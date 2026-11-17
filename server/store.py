"""People, face gallery and conversation storage.

Backend is chosen by DATABASE_URL:
  * empty (default)          -> local SQLite file data/glasses.db  (laptop demo)
  * postgresql://user:pw@... -> Postgres (Supabase, Neon, RDS...)   (cloud, for glasses)
Both use the same schema and code below. Existing SQLite files are upgraded in place.

Per person:
  * attributes: name, relationship, notes (yours, never overwritten by AI), photo
  * profile        - one JSON object enriched by AI after every conversation
                     (key_facts, interests, communication_tips, follow_ups, recall_hint, designation)
  * latest_summary - summary of the most recent conversation
Per conversation: title, summary, topics, mood, key points, follow-ups, full transcript.
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id {ID},
    name TEXT NOT NULL,
    named INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'auto',
    created_at {FLOAT} NOT NULL,
    last_seen {FLOAT},
    conversation_count INTEGER NOT NULL DEFAULT 0,
    profile TEXT NOT NULL DEFAULT '{{}}',
    latest_summary TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS face_embeddings (
    id {ID},
    person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    vec {BLOB} NOT NULL,
    created_at {FLOAT} NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    id {ID},
    person_id INTEGER REFERENCES people(id) ON DELETE SET NULL,
    started_at {FLOAT} NOT NULL,
    ended_at {FLOAT} NOT NULL,
    transcript TEXT NOT NULL,
    summary TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_person ON conversations(person_id, ended_at);
CREATE INDEX IF NOT EXISTS idx_emb_person ON face_embeddings(person_id);
"""

# Columns added after v1 - added automatically to older databases.
MIGRATIONS = {
    "people": {"notes": "TEXT NOT NULL DEFAULT ''", "photo": "{BLOB}"},
    "conversations": {"title": "TEXT NOT NULL DEFAULT ''", "topics": "TEXT NOT NULL DEFAULT '[]'",
                      "mood": "TEXT NOT NULL DEFAULT ''", "key_points": "TEXT NOT NULL DEFAULT '[]'",
                      "follow_ups": "TEXT NOT NULL DEFAULT '[]'", "turn_count": "INTEGER NOT NULL DEFAULT 0"},
}

PROFILE_LISTS = ("key_facts", "interests", "communication_tips", "follow_ups")
PROFILE_TEXT = ("relationship", "recall_hint", "designation")


# ------------------------------------------------------------------ dialect adapter
class DB:
    def __init__(self, url: str):
        self.url = url
        self.pg = url.startswith(("postgres://", "postgresql://"))
        self.tok = ({"ID": "SERIAL PRIMARY KEY", "FLOAT": "DOUBLE PRECISION", "BLOB": "BYTEA"} if self.pg
                    else {"ID": "INTEGER PRIMARY KEY AUTOINCREMENT", "FLOAT": "REAL", "BLOB": "BLOB"})
        self._connect()

    def _connect(self):
        if self.pg:
            import psycopg
            from psycopg.rows import dict_row
            self.conn = psycopg.connect(self.url, row_factory=dict_row, autocommit=True)
        else:
            self.conn = sqlite3.connect(self.url, check_same_thread=False, isolation_level=None)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")

    def sql(self, s: str) -> str:
        return s.replace("?", "%s") if self.pg else s

    def run(self, s: str, params=()) -> list[dict]:
        for attempt in range(2):
            try:
                cur = self.conn.execute(self.sql(s), params)
                if cur.description is None:
                    return []
                return [dict(r) for r in cur.fetchall()]
            except Exception as e:
                # Cloud databases drop idle connections: reconnect once.
                if self.pg and attempt == 0 and "closed" in str(e).lower():
                    self._connect()
                    continue
                raise

    def insert(self, s: str, params=()) -> int:
        return int(self.run(s + " RETURNING id", params)[0]["id"])

    @contextlib.contextmanager
    def transaction(self):
        if self.pg:
            with self.conn.transaction():
                yield
        else:
            self.conn.execute("BEGIN")
            try:
                yield
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    def init_schema(self):
        for stmt in SCHEMA.format(**self.tok).split(";"):
            if stmt.strip():
                self.conn.execute(stmt)
        for table, cols in MIGRATIONS.items():
            existing = self.columns(table)
            for col, typ in cols.items():
                if col not in existing:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ.format(**self.tok)}")

    def columns(self, table: str) -> set[str]:
        if self.pg:
            rows = self.run("SELECT column_name FROM information_schema.columns WHERE table_name=?", (table,))
            return {r["column_name"] for r in rows}
        return {r["name"] for r in self.run(f"PRAGMA table_info({table})")}


# ------------------------------------------------------------------ models
@dataclass
class Person:
    id: int
    name: str
    named: bool
    source: str
    conversation_count: int
    profile: dict = field(default_factory=dict)
    latest_summary: str = ""
    last_seen: float | None = None
    created_at: float | None = None
    notes: str = ""
    has_photo: bool = False

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "named": self.named, "source": self.source,
                "conversation_count": self.conversation_count, "profile": self.profile,
                "latest_summary": self.latest_summary, "last_seen": self.last_seen,
                "first_met": self.created_at, "notes": self.notes, "has_photo": self.has_photo}


PERSON_COLS = ("id, name, named, source, created_at, last_seen, conversation_count, profile, "
               "latest_summary, notes, (photo IS NOT NULL) AS has_photo")


class Store:
    def __init__(self, url: str | None = None):
        self._lock = threading.RLock()
        self.db = DB(url or config.DATABASE_URL or str(config.DB_PATH))
        self.db.init_schema()
        self._reload_gallery()

    @property
    def backend(self) -> str:
        return "postgres" if self.db.pg else "sqlite"

    # ---------------- face gallery (in-memory matrix for fast matching) -------------
    def _reload_gallery(self):
        rows = self.db.run("SELECT person_id, vec FROM face_embeddings")
        if rows:
            self._g_vecs = np.stack([np.frombuffer(bytes(r["vec"]), dtype=np.float32) for r in rows])
            self._g_ids = np.array([r["person_id"] for r in rows], dtype=np.int64)
        else:
            self._g_vecs = np.zeros((0, 128), dtype=np.float32)
            self._g_ids = np.zeros((0,), dtype=np.int64)

    def match(self, emb: np.ndarray) -> tuple[int | None, float]:
        """Best (person_id, cosine similarity). Matching never touches the database,
        so a cloud database adds no delay to live recognition."""
        with self._lock:
            if len(self._g_ids) == 0:
                return None, 0.0
            sims = self._g_vecs @ emb
            i = int(np.argmax(sims))
            best = float(sims[i])
            if best < config.FACE_MATCH_THRESHOLD:
                return None, best
            return int(self._g_ids[i]), best

    def add_embedding(self, person_id: int, emb: np.ndarray, force: bool = False) -> bool:
        """Grow a person's gallery with different-looking views; skip near-duplicates."""
        with self._lock:
            mine = self._g_vecs[self._g_ids == person_id]
            if not force and len(mine):
                if len(mine) >= config.FACE_MAX_EMBEDDINGS or float(np.max(mine @ emb)) > 0.85:
                    return False
            self.db.run("INSERT INTO face_embeddings(person_id, vec, created_at) VALUES (?,?,?)",
                        (person_id, emb.astype(np.float32).tobytes(), time.time()))
            self._g_vecs = np.vstack([self._g_vecs, emb[None, :].astype(np.float32)])
            self._g_ids = np.append(self._g_ids, person_id)
            return True

    # ---------------- people ----------------
    def create_person(self, name: str | None = None, source: str = "auto") -> int:
        with self._lock:
            now = time.time()
            pid = self.db.insert(
                "INSERT INTO people(name, named, source, created_at, last_seen, profile, latest_summary, notes) "
                "VALUES (?,?,?,?,?,?,?,?)", (name or "", 1 if name else 0, source, now, now, "{}", "", ""))
            if not name:
                self.db.run("UPDATE people SET name=? WHERE id=?", (f"Person {pid}", pid))
            return pid

    def get(self, pid: int) -> Person | None:
        rows = self.db.run(f"SELECT {PERSON_COLS} FROM people WHERE id=?", (pid,))
        return _to_person(rows[0]) if rows else None

    def list_people(self) -> list[Person]:
        rows = self.db.run(f"SELECT {PERSON_COLS} FROM people ORDER BY last_seen DESC")
        return [_to_person(r) for r in rows]

    def rename(self, pid: int, name: str):
        self.db.run("UPDATE people SET name=?, named=1 WHERE id=?", (name.strip(), pid))

    def update_person(self, pid: int, *, name: str | None = None, notes: str | None = None,
                      profile: dict | None = None) -> Person | None:
        """Manual edits from the People page. Profile keys are merged, not replaced."""
        p = self.get(pid)
        if not p:
            return None
        if name is not None and name.strip():
            self.rename(pid, name)
        if notes is not None:
            self.db.run("UPDATE people SET notes=? WHERE id=?", (notes, pid))
        if profile:
            merged = dict(p.profile)
            for k, v in profile.items():
                if k in PROFILE_LISTS:
                    merged[k] = [str(x).strip() for x in (v or []) if str(x).strip()][:12]
                elif k in PROFILE_TEXT:
                    merged[k] = str(v or "").strip()
            self.db.run("UPDATE people SET profile=? WHERE id=?", (json.dumps(merged, ensure_ascii=False), pid))
        return self.get(pid)

    def touch(self, pid: int):
        self.db.run("UPDATE people SET last_seen=? WHERE id=?", (time.time(), pid))

    def set_photo(self, pid: int, jpeg: bytes, only_if_missing: bool = True):
        q = "UPDATE people SET photo=? WHERE id=?" + (" AND photo IS NULL" if only_if_missing else "")
        self.db.run(q, (jpeg, pid))

    def photo(self, pid: int) -> bytes | None:
        rows = self.db.run("SELECT photo FROM people WHERE id=?", (pid,))
        return bytes(rows[0]["photo"]) if rows and rows[0]["photo"] is not None else None

    def merge(self, keep_id: int, drop_id: int):
        """Fix a duplicate auto-registration (same person registered twice)."""
        with self._lock, self.db.transaction():
            self.db.run("UPDATE face_embeddings SET person_id=? WHERE person_id=?", (keep_id, drop_id))
            self.db.run("UPDATE conversations SET person_id=? WHERE person_id=?", (keep_id, drop_id))
            n = self.db.run("SELECT COUNT(*) AS n FROM conversations WHERE person_id=?", (keep_id,))[0]["n"]
            self.db.run("UPDATE people SET conversation_count=? WHERE id=?", (n, keep_id))
            self.db.run("UPDATE people SET photo=(SELECT photo FROM people WHERE id=?) "
                        "WHERE id=? AND photo IS NULL", (drop_id, keep_id))
            self.db.run("DELETE FROM people WHERE id=?", (drop_id,))
        with self._lock:
            self._reload_gallery()

    def delete_person(self, pid: int):
        with self._lock, self.db.transaction():
            self.db.run("DELETE FROM face_embeddings WHERE person_id=?", (pid,))
            self.db.run("UPDATE conversations SET person_id=NULL WHERE person_id=?", (pid,))
            self.db.run("DELETE FROM people WHERE id=?", (pid,))
        with self._lock:
            self._reload_gallery()

    # ---------------- conversations ----------------
    def save_conversation(self, pid: int | None, started: float, ended: float,
                          transcript: list[dict], record: dict, new_profile: dict | None) -> int:
        """record: summary, title, topics, mood, key_points, follow_ups (from the AI)."""
        clean = [{k: t[k] for k in ("speaker", "text", "t") if k in t} for t in transcript]
        with self._lock, self.db.transaction():
            cid = self.db.insert(
                "INSERT INTO conversations(person_id, started_at, ended_at, transcript, summary, title, "
                "topics, mood, key_points, follow_ups, turn_count) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (pid, started, ended, json.dumps(clean, ensure_ascii=False), record.get("summary", ""),
                 record.get("title", ""), json.dumps(record.get("topics", []), ensure_ascii=False),
                 record.get("mood", ""), json.dumps(record.get("key_points", []), ensure_ascii=False),
                 json.dumps(record.get("follow_ups", []), ensure_ascii=False), len(clean)))
            if pid is not None:
                self.db.run(
                    "UPDATE people SET latest_summary=?, profile=?, "
                    "conversation_count=conversation_count+1, last_seen=? WHERE id=?",
                    (record.get("summary", ""), json.dumps(new_profile or {}, ensure_ascii=False), ended, pid))
            return cid

    def conversations(self, pid: int | None, limit: int = 50) -> list[dict]:
        cols = "id, person_id, started_at, ended_at, summary, title, topics, mood, key_points, follow_ups, turn_count"
        if pid is None:
            rows = self.db.run(f"SELECT {cols} FROM conversations WHERE person_id IS NULL "
                               "ORDER BY ended_at DESC LIMIT ?", (limit,))
        else:
            rows = self.db.run(f"SELECT {cols} FROM conversations WHERE person_id=? "
                               "ORDER BY ended_at DESC LIMIT ?", (pid, limit))
        return [_to_conv(r) for r in rows]

    def conversation(self, cid: int) -> dict | None:
        rows = self.db.run("SELECT * FROM conversations WHERE id=?", (cid,))
        if not rows:
            return None
        c = _to_conv(rows[0])
        c["transcript"] = json.loads(rows[0]["transcript"] or "[]")
        return c

    def assign_conversation(self, cid: int, pid: int | None):
        with self._lock, self.db.transaction():
            old = self.db.run("SELECT person_id FROM conversations WHERE id=?", (cid,))
            if not old:
                return
            self.db.run("UPDATE conversations SET person_id=? WHERE id=?", (pid, cid))
            for p in {old[0]["person_id"], pid} - {None}:
                n = self.db.run("SELECT COUNT(*) AS n FROM conversations WHERE person_id=?", (p,))[0]["n"]
                self.db.run("UPDATE people SET conversation_count=? WHERE id=?", (n, p))

    def delete_conversation(self, cid: int):
        old = self.db.run("SELECT person_id FROM conversations WHERE id=?", (cid,))
        self.db.run("DELETE FROM conversations WHERE id=?", (cid,))
        if old and old[0]["person_id"] is not None:
            p = old[0]["person_id"]
            n = self.db.run("SELECT COUNT(*) AS n FROM conversations WHERE person_id=?", (p,))[0]["n"]
            self.db.run("UPDATE people SET conversation_count=? WHERE id=?", (n, p))

    def export(self) -> dict:
        """Everything except face embeddings and photos, as plain JSON."""
        people = []
        for p in self.list_people():
            d = p.public()
            d["conversations"] = [self.conversation(c["id"]) for c in self.conversations(p.id, 1000)]
            people.append(d)
        return {"exported_at": time.time(), "people": people,
                "unassigned_conversations": [self.conversation(c["id"]) for c in self.conversations(None, 1000)]}


def _to_person(r: dict) -> Person:
    return Person(id=r["id"], name=r["name"], named=bool(r["named"]), source=r["source"],
                  conversation_count=r["conversation_count"], profile=json.loads(r["profile"] or "{}"),
                  latest_summary=r["latest_summary"] or "", last_seen=r["last_seen"],
                  created_at=r["created_at"], notes=r.get("notes") or "", has_photo=bool(r.get("has_photo")))


def _to_conv(r: dict) -> dict:
    out = {k: r[k] for k in ("id", "person_id", "started_at", "ended_at", "summary", "title", "mood", "turn_count")}
    for k in ("topics", "key_points", "follow_ups"):
        out[k] = json.loads(r[k] or "[]")
    out["duration_s"] = round((r["ended_at"] or 0) - (r["started_at"] or 0))
    return out
