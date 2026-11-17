"""Hardware-free tests: `python -m pytest -q` from the project root.

Database tests run on SQLite, and also on Postgres when TEST_DATABASE_URL is set:
    TEST_DATABASE_URL=postgresql://user@localhost:5432/postgres python -m pytest -q
"""
import asyncio
import json
import os
import sqlite3
import tempfile

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())

import numpy as np
import pytest

from server import audio, config, llm, prompts
from server.face import Face
from server.identity import IdentityTracker
from server.store import Store

rng = np.random.default_rng(0)
PG_URL = os.getenv("TEST_DATABASE_URL")


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def near(base, noise=0.15):
    return unit(base + rng.normal(0, noise / np.sqrt(128), 128))


@pytest.fixture(params=["sqlite"] + (["postgres"] if PG_URL else []))
def store(request, tmp_path):
    if request.param == "sqlite":
        return Store(str(tmp_path / "t.db"))
    import psycopg
    with psycopg.connect(PG_URL, autocommit=True) as c:
        c.execute("DROP TABLE IF EXISTS conversations, face_embeddings, people CASCADE")
    return Store(PG_URL)


RECORD = {"title": "Said hi", "summary": "said hi", "topics": ["greeting"], "mood": "cheerful",
          "key_points": ["They said hi"], "follow_ups": ["Send photos"]}


# ------------------------------------------------------------ store / matching
def test_match_and_gallery_growth(store):
    a, b = unit(rng.normal(size=128)), unit(rng.normal(size=128))
    pa, pb = store.create_person("Asha"), store.create_person()
    store.add_embedding(pa, a, force=True); store.add_embedding(pb, b, force=True)
    assert store.match(near(a))[0] == pa
    assert store.match(near(b))[0] == pb
    assert store.match(unit(rng.normal(size=128)))[0] is None
    assert store.get(pb).name == f"Person {pb}" and not store.get(pb).named
    assert not store.add_embedding(pa, a)          # near-duplicate is skipped
    # gallery survives a restart (reloaded from the database)
    store._reload_gallery()
    assert store.match(near(a))[0] == pa


def test_conversation_record_and_profile(store):
    pid = store.create_person("Ravi")
    turns = [{"speaker": "them", "text": "hi", "t": 1.0, "db": -30, "f0": 200, "source": "auto"}]
    cid = store.save_conversation(pid, 0, 95, turns, RECORD, {"recall_hint": "x", "key_facts": ["a"]})
    p = store.get(pid)
    assert p.latest_summary == "said hi" and p.profile["recall_hint"] == "x" and p.conversation_count == 1
    c = store.conversation(cid)
    assert c["title"] == "Said hi" and c["topics"] == ["greeting"] and c["mood"] == "cheerful"
    assert c["follow_ups"] == ["Send photos"] and c["duration_s"] == 95 and c["turn_count"] == 1
    assert c["transcript"] == [{"speaker": "them", "text": "hi", "t": 1.0}]   # internal audio features not stored
    assert store.conversations(pid)[0]["key_points"] == ["They said hi"]


def test_manual_edits_merge_into_profile(store):
    pid = store.create_person()
    store.save_conversation(pid, 0, 1, [], RECORD, {"interests": ["cricket"], "recall_hint": "neighbour"})
    p = store.update_person(pid, name="Meena", notes="Prefers texting",
                            profile={"relationship": "Neighbour", "follow_ups": ["Ask about exam", " "],
                                     "bogus_key": "ignored"})
    assert p.name == "Meena" and p.named and p.notes == "Prefers texting"
    assert p.profile["interests"] == ["cricket"]                    # untouched keys kept
    assert p.profile["follow_ups"] == ["Ask about exam"] and "bogus_key" not in p.profile


def test_photo_merge_assign_delete(store):
    a = unit(rng.normal(size=128))
    p1, p2 = store.create_person(), store.create_person()
    store.add_embedding(p1, a, force=True); store.add_embedding(p2, near(a), force=True)
    store.set_photo(p2, b"\xff\xd8jpeg")
    c = store.save_conversation(p2, 0, 1, [], RECORD, {})
    store.merge(p1, p2)
    assert store.get(p2) is None and store.get(p1).conversation_count == 1
    assert store.photo(p1) == b"\xff\xd8jpeg" and store.get(p1).has_photo
    loose = store.save_conversation(None, 0, 1, [], RECORD, None)
    assert [x["id"] for x in store.conversations(None)] == [loose]
    store.assign_conversation(loose, p1)
    assert store.get(p1).conversation_count == 2 and store.conversations(None) == []
    store.delete_conversation(c)
    assert store.get(p1).conversation_count == 1
    store.delete_person(p1)
    assert store.list_people() == [] and store.match(a)[0] is None
    assert store.conversations(None)[0]["id"] == loose            # history kept, unassigned


def test_export_has_no_biometrics(store):
    pid = store.create_person("Asha")
    store.add_embedding(pid, unit(rng.normal(size=128)), force=True)
    store.set_photo(pid, b"jpeg")
    store.save_conversation(pid, 0, 1, [{"speaker": "me", "text": "hello"}], RECORD, {})
    data = json.dumps(store.export())
    assert "Asha" in data and "hello" in data and '"vec"' not in data and '"photo"' not in data


def test_old_v1_database_is_upgraded(tmp_path):
    path = tmp_path / "old.db"
    c = sqlite3.connect(path)
    c.executescript("""
        CREATE TABLE people (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
          named INTEGER NOT NULL DEFAULT 0, source TEXT NOT NULL DEFAULT 'auto', created_at REAL NOT NULL,
          last_seen REAL, conversation_count INTEGER NOT NULL DEFAULT 0, profile TEXT NOT NULL DEFAULT '{}',
          latest_summary TEXT NOT NULL DEFAULT '');
        CREATE TABLE conversations (id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER,
          started_at REAL NOT NULL, ended_at REAL NOT NULL, transcript TEXT NOT NULL, summary TEXT NOT NULL);
        INSERT INTO people(name, named, created_at) VALUES ('Old Friend', 1, 1.0);
        INSERT INTO conversations(person_id, started_at, ended_at, transcript, summary)
          VALUES (1, 0, 60, '[]', 'old summary');""")
    c.commit(); c.close()
    s = Store(str(path))
    assert s.get(1).name == "Old Friend" and s.get(1).notes == ""
    assert s.conversations(1)[0]["summary"] == "old summary" and s.conversations(1)[0]["topics"] == []


# ------------------------------------------------------------ identity / auto-register
def test_auto_register_needs_consistent_sightings(store):
    tr = IdentityTracker(store, auto_register=True)
    face = unit(rng.normal(size=128))
    crop = np.full((112, 112, 3), 120, np.uint8)
    t0 = 1000.0
    states = [tr.update([Face((0, 0, 100, 100), .9, near(face), crop)], now=t0 + i * 0.4) for i in range(6)]
    assert any(s.just_registered for s in states)
    pid = next(s.person_id for s in states if s.just_registered)
    assert store.get(pid).has_photo
    tr2 = IdentityTracker(store)
    tr2.update([Face((0, 0, 100, 100), .9, near(face))], now=t0 + 10)
    s2 = tr2.update([Face((0, 0, 100, 100), .9, near(face))], now=t0 + 10.4)
    assert s2.person_id == pid and not s2.just_registered
    assert len(store.list_people()) == 1


def test_single_glimpse_does_not_register(store):
    tr = IdentityTracker(store)
    tr.update([Face((0, 0, 100, 100), .9, unit(rng.normal(size=128)))], now=5.0)
    assert store.list_people() == []


# ------------------------------------------------------------ audio helpers
def voice(seconds, amp, f0=140.0):
    """Speech-like signal: harmonic stack with pitch wobble and syllable-rate modulation."""
    t = np.arange(int(seconds * 16000)) / 16000
    pitch = f0 * (1 + 0.08 * np.sin(2 * np.pi * 3 * t))
    phase = 2 * np.pi * np.cumsum(pitch) / 16000
    sig = sum(np.sin(k * phase) / k for k in range(1, 12))
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)
    return (sig * env / 3 * amp * 32767).astype(np.int16)


def silence(seconds):
    return (rng.normal(0, 30, int(seconds * 16000))).astype(np.int16)


def run(det, *chunks):
    events = []
    for c in chunks:
        b = c.tobytes()
        for i in range(0, len(b), 1280):
            events += det.feed(b[i:i + 1280])
    return events


def utterances(ev):
    return [e for e in ev if e["type"] == "utterance"]


def calibrated(f0=120.0, amp=0.3, mode="auto"):
    det = audio.TurnDetector(mode=mode)
    det.start_calibration()
    run(det, voice(3.0, amp, f0), silence(1.0))
    assert det.finish_calibration() is not None
    return det


# ------------------------------------------------------------ pitch + speaker detection
def test_pitch_estimate():
    for f0 in (110.0, 210.0):
        frames = [f for f in voice(1.0, 0.3, f0).reshape(-1, 320)]
        est = audio.utterance_f0(frames, [True] * len(frames))
        assert abs(audio.semitones(est, f0)) < 1.5


def test_calibration_records_level_and_pitch():
    det = calibrated(f0=120)
    me = det.model.me
    assert -30 < me["db"] < 0 and abs(audio.semitones(me["f0"], 120)) < 1.5


def test_uncalibrated_auto_assumes_partner():
    det = audio.TurnDetector()
    u = utterances(run(det, silence(.4), voice(1.2, .3, 120), silence(1.0)))
    assert u[0]["speaker"] == "them" and u[0]["source"] == "auto"


def test_auto_same_loudness_told_apart_by_pitch():
    """Laptop case: both people equally loud, different voices."""
    det = calibrated(f0=120, amp=0.3)
    me = utterances(run(det, silence(.4), voice(1.5, .3, 125), silence(1.0)))
    them = utterances(run(det, silence(.4), voice(1.5, .3, 230), silence(2.0)))
    assert me[0]["speaker"] == "me" and them[0]["speaker"] == "them"


def test_auto_glasses_case_partner_quieter():
    det = calibrated(f0=120, amp=0.4)
    them = utterances(run(det, silence(.4), voice(1.5, .03, 125), silence(1.0)))
    assert them[0]["speaker"] == "them"            # same pitch range, but far quieter


def test_manual_mode_hold_to_talk():
    det = audio.TurnDetector(mode="manual")
    det.press(True)
    a = utterances(run(det, silence(.4), voice(1.2, .3, 230)))
    a += utterances(run(det, silence(1.0)))
    det.press(False)
    b = utterances(run(det, silence(.4), voice(1.2, .3, 120), silence(1.0)))
    assert a[0]["speaker"] == "me" and a[0]["source"] == "button"
    assert b[0]["speaker"] == "them" and b[0]["source"] == "button"
    assert det.model.calibrated                    # button use taught the auto model my voice


def test_button_overrides_auto_and_overlap_on_press():
    det = calibrated(f0=120)
    det.feed(silence(.4).tobytes()); det.feed(voice(0.5, .3, 230).tobytes())
    assert det.in_speech and det.start_speaker == "them"
    assert [e["type"] for e in det.press(True)] == ["overlap"]
    u = utterances(run(det, voice(1.0, .3, 230), silence(1.0)))
    assert u[0]["speaker"] == "me" and u[0]["source"] == "button"


def test_correction_teaches_model():
    det = audio.TurnDetector()
    det.correct(-14.0, 180.0, "me")
    assert det.model.calibrated and det.model.me["f0"] == 180.0


def test_your_turn_after_partner_pause():
    det = calibrated(f0=120)
    ev = run(det, silence(0.4), voice(1.5, 0.3, 230), silence(2.0))
    kinds = [e["type"] for e in ev]
    assert "your_turn" in kinds and kinds.index("your_turn") > kinds.index("utterance")
    assert det.floor_open
    assert "long_pause" not in kinds


def test_long_pause_follows_the_usual_pause():
    det = calibrated(f0=120)
    ev = run(det, silence(0.4), voice(1.2, 0.3, 230), silence(1.6))
    assert "your_turn" in [e["type"] for e in ev]
    assert not det.long_pause_open
    ev += run(det, silence(2.2))
    kinds = [e["type"] for e in ev]
    assert det.long_pause_open
    assert kinds.index("long_pause") > kinds.index("your_turn")


def test_short_noise_does_not_close_an_open_floor():
    """A cough after 'your turn' is not a new turn, and it does not revoke the pause."""
    det = calibrated(f0=120)
    run(det, silence(0.4), voice(1.2, 0.3, 230), silence(1.6))
    assert det.floor_open
    ev = run(det, voice(0.25, 0.3, 230), silence(0.9))
    assert not utterances(ev)
    assert any(e["type"] == "false_start" for e in ev)
    assert det.floor_open


def test_floor_stays_closed_until_your_turn_pause():
    det = calibrated(f0=120)
    run(det, silence(0.4), voice(1.2, 0.3, 230), silence(0.9))
    assert not det.floor_open
    run(det, silence(0.5))
    assert det.floor_open


def test_short_click_ignored():
    det = audio.TurnDetector()
    assert not utterances(run(det, silence(0.3), voice(0.15, 0.5), silence(1.5)))


# ------------------------------------------------------------ LLM output shaping
def test_cue_count_capped_not_padded(monkeypatch):
    """A short list is kept. The model is not called again to invent the missing lines."""
    calls = []

    async def fake(system, user, **kw):
        if system == prompts.CHECK_SYSTEM:
            return {"keep": [0]}
        calls.append(user)
        return {"replies": ["That sounds really great when is the marathon happening"],
                "social": "They're excited", "emotion": "excited"}

    monkeypatch.setattr(llm, "chat_json", fake)
    turns = [{"speaker": "them", "text": "I got into the marathon!", "conf": 0.9}]
    out = asyncio.run(llm.live_cues(None, turns, {"num_replies": 3, "max_words": 6},
                                    moment="They finished a statement and left a pause."))
    assert out["replies"] == ["That sounds really great when is the marathon happening"]
    assert "…" not in out["replies"][0]
    assert len(calls) == 1
    assert "I got into the marathon!" in calls[0]
    assert "left a pause" in calls[0]


def test_checker_drops_invented_fact_and_keeps_relevant_reply(monkeypatch):
    """A made-up answer goes. A helpful question stays, even in new words."""
    calls = []

    async def fake(system, user, **kw):
        calls.append((system, user))
        if system == prompts.CHECK_SYSTEM:
            return {"keep": [1, 9]}
        return {"replies": ["I'm from Delhi", "What about you?"],
                "social": "Answer in your own words", "emotion": "curious"}

    monkeypatch.setattr(llm, "chat_json", fake)

    class P:
        name = "Ravi"
        named = True
        notes = ""
        latest_summary = ""
        profile = {"key_facts": ["dog is Max"]}

    turns = [{"speaker": "them", "text": "Where are you from?", "conf": 0.9}]
    out = asyncio.run(llm.live_cues(P(), turns, {"num_replies": 2, "max_words": 8}))
    assert out["replies"] == ["Good question, let me think", "What about you?"]
    assert out["social"] == "Answer in your own words" and out["emotion"] == "curious"
    assert len(calls) == 2
    assert calls[0][0] != prompts.CHECK_SYSTEM and calls[1][0] == prompts.CHECK_SYSTEM
    check_user = calls[1][1]
    assert "Where are you from?" in check_user
    assert "I'm from Delhi" in check_user and "What about you?" in check_user
    assert "dog is Max" in check_user


def test_checker_preserves_suggester_order(monkeypatch):
    async def fake(system, user, **kw):
        if system == prompts.CHECK_SYSTEM:
            return {"keep": [1, 9, "0", True]}
        return {"replies": ["Sounds exciting", "How are you feeling?"],
                "social": None, "emotion": None}

    monkeypatch.setattr(llm, "chat_json", fake)
    turns = [{"speaker": "them", "text": "I got into the marathon!", "conf": 0.9}]
    out = asyncio.run(llm.live_cues(None, turns, {"num_replies": 2}))
    assert out["replies"] == ["Sounds exciting", "How are you feeling?"]


def test_checker_keeps_paraphrase_missing_from_transcript(monkeypatch):
    reply = "Congrats on the race, how are you feeling about it?"
    seen = {}

    async def fake(system, user, **kw):
        if system == prompts.CHECK_SYSTEM:
            seen["check"] = user
            seen["temperature"] = kw.get("temperature")
            return {"keep": [0]}
        return {"replies": [reply], "social": None, "emotion": None}

    monkeypatch.setattr(llm, "chat_json", fake)
    turns = [{"speaker": "them", "text": "I got into the marathon!", "conf": 0.9}]
    out = asyncio.run(llm.live_cues(None, turns, {"num_replies": 2, "max_words": 12}))
    assert out["replies"] == [reply]
    assert "race" not in turns[0]["text"]
    assert "I got into the marathon!" in seen["check"] and reply in seen["check"]
    assert seen["temperature"] == 0


def test_checker_failure_uses_generic_replies(monkeypatch):
    """An unverified line is not shown. A generic one takes its place."""
    async def exploded(system, user, **kw):
        if system == prompts.CHECK_SYSTEM:
            raise llm.LLMError("checker down")
        return {"replies": ["I'm from Delhi"], "social": "They asked where you live", "emotion": None}

    monkeypatch.setattr(llm, "chat_json", exploded)
    turns = [{"speaker": "them", "text": "Where are you from?", "conf": 0.9}]
    out = asyncio.run(llm.live_cues(None, turns, {"num_replies": 2}))
    assert out["replies"] == ["Good question, let me think"]
    assert "Delhi" not in out["replies"][0]
    assert out["social"] == "They asked where you live"

    async def missing_keep(system, user, **kw):
        if system == prompts.CHECK_SYSTEM:
            return {"ok": True}
        return {"replies": ["I'm from Delhi"], "social": None, "emotion": None}

    monkeypatch.setattr(llm, "chat_json", missing_keep)
    out = asyncio.run(llm.live_cues(None, turns, {"num_replies": 2}))
    assert out == {
        "replies": ["Good question, let me think"],
        "social": None, "emotion": None,
    }


def test_every_dropped_reply_becomes_generic(monkeypatch):
    async def fake(system, user, **kw):
        if system == prompts.CHECK_SYSTEM:
            return {"keep": []}
        return {"replies": ["I live in Pune", "My dog is Max"], "social": None, "emotion": None}

    monkeypatch.setattr(llm, "chat_json", fake)
    turns = [{"speaker": "them", "text": "I got into the marathon!", "conf": 0.9}]
    out = asyncio.run(llm.live_cues(None, turns, {"num_replies": 2}))
    assert out["replies"] == ["Tell me more about that", "How are you feeling about it?"]
    assert "Pune" not in " ".join(out["replies"]) and "Max" not in " ".join(out["replies"])


def test_opening_line_rejected_by_checker_becomes_hello(monkeypatch):
    async def fake(system, user, **kw):
        if system == prompts.CHECK_SYSTEM:
            assert "Pune" in user
            return {"keep": []}
        return {"replies": ["Hey, how's the new job in Pune?"]}

    monkeypatch.setattr(llm, "chat_json", fake)

    class Known:
        name = "Ravi"
        named = True
        notes = ""
        latest_summary = "talked about the exam"
        conversation_count = 1
        profile = {"follow_ups": ["Ask about the exam"]}

    assert asyncio.run(llm.opening_line(Known(), {"max_words": 10})) == ["Hey, good to see you"]


def test_empty_suggester_gets_generic_replies(monkeypatch):
    """The model used to return nothing rather than risk a fact. The wearer still gets a line."""
    calls = []

    async def fake(system, user, **kw):
        calls.append(user)
        return {"replies": [], "social": None, "emotion": None}

    monkeypatch.setattr(llm, "chat_json", fake)
    turns = [{"speaker": "them", "text": "I went to the market with my sister.", "conf": 0.9}]
    out = asyncio.run(llm.live_cues(None, turns, {"num_replies": 2, "max_words": 8}))
    assert out["replies"] == ["Tell me more about that", "How are you feeling about it?"]
    assert out["social"] is None and out["emotion"] is None
    assert len(calls) == 1


def test_empty_icebreaker_gets_a_generic_question(monkeypatch):
    """A stalled conversation still gets something to say when the model writes nothing."""
    calls = []

    async def fake(system, user, **kw):
        calls.append(user)
        return {"replies": []}

    monkeypatch.setattr(llm, "chat_json", fake)
    turns = [{"speaker": "them", "text": "I went to the market with my sister.", "conf": 0.9}]
    out = asyncio.run(llm.icebreaker(None, turns, {"num_replies": 2, "max_words": 10, "social": True}))
    assert out["kind"] == "icebreaker"
    assert out["replies"] == ["How's your week been?"]
    assert out["social"] == "Try a new topic"
    assert len(calls) == 1


def test_disabled_cues_skip_llm(monkeypatch):
    async def boom(*a, **k):
        raise AssertionError("should not call")
    monkeypatch.setattr(llm, "chat_json", boom)
    out = asyncio.run(llm.live_cues(None, [], {"num_replies": 0, "social": False, "emotion": False}))
    assert out["replies"] == []


def test_summary_record_shape(monkeypatch):
    async def fake(system, user, **kw):
        assert "Prefers texting" in user                     # wearer's notes reach the AI
        return {"title": "New job", "summary": "Talked about job.", "topics": "work", "mood": "happy",
                "key_points": ["Starts Monday"], "conversation_follow_ups": [], "name": "Ravi",
                "profile": {"key_facts": "works in Pune", "recall_hint": "Ravi – new job"}}
    monkeypatch.setattr(llm, "chat_json", fake)

    class P:
        profile = {"relationship": "Neighbour", "designation": "Teacher"}
        latest_summary = ""; notes = "Prefers texting"
    out = asyncio.run(llm.summarize(P(), [{"speaker": "them", "text": "hi", "conf": 1}]))
    assert out["record"]["topics"] == ["work"] and out["record"]["title"] == "New job"
    assert out["profile"]["key_facts"] == ["works in Pune"]
    assert out["profile"]["relationship"] == "Neighbour"     # kept when AI leaves it blank
    assert out["profile"]["designation"] == "Teacher"        # kept when AI omits it
    assert out["name"] == "Ravi"


def test_summary_stores_returned_designation(monkeypatch):
    async def fake(system, user, **kw):
        return {"title": "New job", "summary": "Talked about job.", "topics": [], "mood": "",
                "key_points": [], "conversation_follow_ups": [], "name": None,
                "profile": {"designation": "Math teacher at the school on the hill with a very long extra clause",
                            "relationship": ""}}
    monkeypatch.setattr(llm, "chat_json", fake)

    class P:
        profile = {"relationship": "Neighbour", "designation": "Teacher"}
        latest_summary = ""; notes = ""
    out = asyncio.run(llm.summarize(P(), [{"speaker": "them", "text": "hi", "conf": 1}]))
    assert out["profile"]["relationship"] == "Neighbour"
    assert out["profile"]["designation"] == "Math teacher at the school on the hill with a very long extr"


def test_person_brief_keeps_the_full_profile():
    class P:
        name = "Ravi"
        named = True
        notes = "allergic to peanuts"
        latest_summary = "talked about the dog Max"
        profile = {
            "relationship": "Neighbour", "designation": "Teacher",
            "key_facts": ["dog is Max"], "follow_ups": ["Ask about the exam"],
        }
    text = llm.person_brief(P())
    assert "Neighbour" in text and "Teacher" in text
    assert "dog is Max" in text and "allergic to peanuts" in text
    assert "talked about the dog Max" in text and "Ask about the exam" in text


def test_opening_line_uses_memory_and_stays_single(monkeypatch):
    calls = []

    async def fake(system, user, **kw):
        if system == prompts.CHECK_SYSTEM:
            return {"keep": [0]}
        calls.append(user)
        return {"replies": ["Hey, how did the exam go?", "second line"]}

    monkeypatch.setattr(llm, "chat_json", fake)

    class Known:
        name = "Ravi"
        named = True
        notes = "allergic to peanuts"
        latest_summary = "talked about the exam"
        conversation_count = 1
        profile = {
            "relationship": "Neighbour", "designation": "Teacher",
            "key_facts": ["dog is Max"], "follow_ups": ["Ask about the exam"],
        }

    out = asyncio.run(llm.opening_line(Known(), {"max_words": 6}))
    assert out == ["Hey, how did the exam go?"]
    assert "…" not in out[0]
    assert len(calls) == 1
    brief = calls[0]
    assert "Bring up this first: Ask about the exam" in brief
    assert "dog is Max" in brief and "allergic to peanuts" in brief
    assert "talked about the exam" in brief and "Teacher" in brief

    class FactsOnly:
        name = "Ravi"
        named = True
        notes = "allergic to peanuts"
        latest_summary = ""
        conversation_count = 0
        profile = {"key_facts": ["dog is Max"], "relationship": "Neighbour"}

    again = []

    async def boom(system, user, **kw):
        again.append(user)
        return {"replies": ["How is Max?"]}

    monkeypatch.setattr(llm, "chat_json", boom)
    assert asyncio.run(llm.opening_line(FactsOnly(), {"max_words": 8})) == []
    assert again == []


def test_announce_card_hides_facts_and_asks_once(store, monkeypatch):
    from server import session as S

    pid = store.create_person("Ravi")
    store.update_person(pid, notes="allergic to peanuts", profile={
        "relationship": "Neighbour", "designation": "Teacher",
        "recall_hint": "dog is Max", "key_facts": ["dog is Max"],
        "communication_tips": ["speak slowly"], "follow_ups": ["Ask about the exam"],
    })
    asked = []

    async def fake_open(person, settings):
        asked.append(llm.person_brief(person))
        return ["How did the exam go?"]

    monkeypatch.setattr(S.llm, "opening_line", fake_open)

    async def main():
        out = []

        async def send(m):
            out.append(m)

        s = S.DeviceSession(send, store, None, None, {})
        await s._announce(pid, False, opener=True)
        if s._opener_task:
            await s._opener_task
        await s._announce(pid, False)          # rename path: no second starter
        await s.close()
        return out

    out = asyncio.run(main())
    person = next(m for m in out if m["type"] == "person")
    blob = json.dumps(person).lower()
    assert person["relationship"] == "Neighbour" and person["designation"] == "Teacher"
    assert "hint" not in person and "facts" not in person and "follow_up" not in person
    assert "max" not in blob and "peanut" not in blob and "exam" not in blob and "slowly" not in blob
    assert len(asked) == 1
    assert "dog is Max" in asked[0] and "allergic to peanuts" in asked[0]
    starters = [m for m in out if m.get("type") == "cues" and m.get("replies")]
    assert starters[0]["replies"] == ["How did the exam go?"]
    assert starters[0]["social"] is None and starters[0]["emotion"] is None


def test_announce_skips_opener_without_memory(store, monkeypatch):
    from server import session as S

    pid = store.create_person("Ravi")
    store.update_person(pid, notes="allergic to peanuts", profile={
        "relationship": "Neighbour", "key_facts": ["dog is Max"],
    })

    async def boom(person, settings):
        raise AssertionError("opener should not be requested")

    monkeypatch.setattr(S.llm, "opening_line", boom)

    async def main():
        out = []

        async def send(m):
            out.append(m)

        s = S.DeviceSession(send, store, None, None, {})
        await s._announce(pid, False, opener=True)
        assert s._opener_task is None
        await s.close()
        return out

    out = asyncio.run(main())
    person = next(m for m in out if m["type"] == "person")
    assert person["relationship"] == "Neighbour"
    assert "dog is Max" not in json.dumps(person)
    assert not any(m.get("type") == "cues" and m.get("replies") for m in out)


# ------------------------------------------------------------ when to suggest
def test_suggestion_expires_when_the_moment_passes():
    from server.cues import suggestion_expired, used_suggestion

    replies = ["Welcome! Where from?", "How's the move going?"]
    assert used_suggestion("Where did you move from?", replies)
    assert used_suggestion("nice", ["That's nice of you"])
    assert not used_suggestion("yeah", replies)
    assert suggestion_expired({"text": "Where from?"}, replies)
    assert suggestion_expired({"text": "What time is the bus?"}, replies)   # different, topic moved
    assert suggestion_expired({"text": "I start the new job on monday"}, [])
    assert not suggestion_expired({"text": "yeah"}, replies)
    assert not suggestion_expired({"text": "oh okay"}, replies)
    assert not suggestion_expired({"text": "cough"}, replies)
    assert not suggestion_expired({"text": "ahem"}, replies)
    assert not suggestion_expired({"text": "*cough*"}, replies)
    assert not suggestion_expired({"text": "The packing took all week"}, replies)
    # Same subject, said a different way. The anchor is the line we answered.
    assert not suggestion_expired(
        {"text": "He helped me carry the boxes inside"},
        replies, anchor="I just moved here last week")
    # A cough long enough to be transcribed can still be a short noise.
    assert not suggestion_expired(
        {"text": "I start the new job on monday", "voiced_ms": 450}, replies)
    # Same subject, answered in the wearer's own words. Overlap keeps it for them.
    own = {"text": "I am doing pretty well"}
    anchor = "How are you doing today?"
    assert not suggestion_expired(own, replies, anchor=anchor)
    assert suggestion_expired(own, replies, anchor=anchor, wearer=True)
    assert not suggestion_expired({"text": "yeah"}, replies, anchor=anchor, wearer=True)
    assert not suggestion_expired({"text": "umm"}, replies, anchor=anchor, wearer=True)
    assert not suggestion_expired(
        {"text": "cough", "voiced_ms": 400}, replies, anchor=anchor, wearer=True)


def test_wearer_hesitation_is_umm_not_agreement():
    from server.cues import hesitation_hits, wearer_hesitating

    assert wearer_hesitating("Umm..")
    assert wearer_hesitating("Ah...")
    assert wearer_hesitating("uh")
    assert wearer_hesitating("uhm")
    assert wearer_hesitating("eh")
    assert wearer_hesitating("hum")
    assert wearer_hesitating("erm")
    assert wearer_hesitating("I don't know")
    assert hesitation_hits("umm") == 1
    assert hesitation_hits("umm ah") == 2
    assert hesitation_hits("I don't know") >= 2
    assert hesitation_hits("yeah") == 0
    assert wearer_hesitating("um so")
    assert wearer_hesitating("um, I don't know")
    assert not wearer_hesitating("uh huh")
    assert not wearer_hesitating("uh-huh")
    assert not wearer_hesitating("mhm")
    assert not wearer_hesitating("yeah")
    assert not wearer_hesitating("ahem")
    assert not wearer_hesitating("I went to the market with my sister")


def test_cue_moment_only_on_real_openings():
    from server.cues import cue_decision, needs_repeat

    def them(text, conf=0.9):
        return {"speaker": "them", "text": text, "conf": conf}

    assert cue_decision(them("How are you?"))[0] == "question"
    assert cue_decision(them("What's up?"))[0] == "question"
    assert cue_decision(them("I got the job"))[0] == "statement"
    assert cue_decision(them("Hi, I'm Ravi, I just moved here."))[0] == "statement"
    assert cue_decision(them("yeah"))[0] is None
    assert cue_decision(them("cough"))[0] is None
    assert cue_decision(them("ahem"))[0] is None
    assert cue_decision(them("ok cool"))[0] is None
    assert cue_decision(them("Right?"))[0] is None
    assert cue_decision(them("and then we went to the"))[0] is None
    assert cue_decision(them("How are you?", conf=0.2))[0] is None
    assert cue_decision({"speaker": "me", "text": "How are you?", "conf": 0.9})[0] is None
    assert needs_repeat(them("I just moved here last week", conf=0.1))
    assert not needs_repeat(them("yeah", conf=0.1))


# ------------------------------------------------------------ full session pipeline
def _session_fakes(monkeypatch):
    from server import session as S

    class FakeSTT:
        def __init__(self):
            self.n = 0

        def transcribe(self, audio_, context=""):
            self.n += 1
            return ("Hi, I'm Ravi, I just moved here." if self.n == 1 else "Nice to meet you Ravi"), 0.9

    async def fake_cues(person, turns, settings, moment=""):
        return {"replies": ["Welcome! Where from?", "How's it going?"], "social": None, "emotion": "friendly"}

    async def fake_summary(person, turns):
        return {"record": {**RECORD, "title": "Met Ravi", "summary": "Ravi just moved here."}, "name": "Ravi",
                "profile": {"key_facts": ["just moved here"], "follow_ups": ["how settling in"],
                            "recall_hint": "Ravi – new neighbour, just moved in"}}

    monkeypatch.setattr(S.llm, "live_cues", fake_cues)
    monkeypatch.setattr(S.llm, "summarize", fake_summary)
    monkeypatch.setattr(S.config, "CUE_MIN_INTERVAL_S", 0)
    # These tests still mean "once the line has ended". The product waits 5s / 8s.
    monkeypatch.setattr(S.config, "CUE_REPLY_AFTER_S", 1.2)
    monkeypatch.setattr(S.config, "CUE_ICEBREAKER_AFTER_S", 3600.0)
    return S, FakeSTT


def test_session_end_to_end(store, monkeypatch):
    S, FakeSTT = _session_fakes(monkeypatch)
    profile = calibrated(f0=120).model.to_dict()

    async def main():
        out = []

        async def send(m):
            out.append(m)

        pid = store.create_person()
        s = S.DeviceSession(send, store, None, FakeSTT(), {"voice_profile": profile})
        s.conv.person_frames[pid] += 3
        await s.on_audio(np.concatenate([silence(.4), voice(1.2, .3, 230), silence(1.5)]).tobytes())
        await asyncio.sleep(0.3)   # suggestion lands during their pause, before the wearer talks
        await s.on_audio(np.concatenate([voice(1.0, .3, 120), silence(1.0)]).tobytes())
        await asyncio.sleep(0.2)
        await s.end_conversation()
        await s.close()
        return pid, out

    pid, out = asyncio.run(main())
    tr = [m for m in out if m["type"] == "transcript"]
    assert [t["speaker"] for t in tr] == ["them", "me"]
    assert any(m["type"] == "cues" and m.get("replies") for m in out)
    assert "your_turn" in [m.get("state") for m in out]
    assert any(m["type"] == "voice_profile" for m in out)
    saved = next(m for m in out if m["type"] == "conversation_saved")
    assert saved["title"] == "Met Ravi"
    p = store.get(pid)
    assert p.name == "Ravi" and p.latest_summary == "Ravi just moved here."
    c = store.conversation(saved["conversation_id"])
    assert c["title"] == "Met Ravi" and [t["speaker"] for t in c["transcript"]] == ["them", "me"]


def test_session_manual_mode_and_fix_speaker(store, monkeypatch):
    S, FakeSTT = _session_fakes(monkeypatch)

    async def main():
        out = []

        async def send(m):
            out.append(m)

        s = S.DeviceSession(send, store, None, FakeSTT(), {"speaker_mode": "manual"})
        await s.control({"type": "ptt", "down": True})
        await s.on_audio(np.concatenate([silence(.4), voice(1.2, .3, 120), silence(1.0)]).tobytes())
        await s.control({"type": "ptt", "down": False})
        await asyncio.sleep(0.2)
        await s.control({"type": "fix_speaker", "index": 0})      # wearer says: that was them
        await asyncio.sleep(0.2)
        await s.close()
        return s, out

    s, out = asyncio.run(main())
    tr = next(m for m in out if m["type"] == "transcript")
    assert tr["speaker"] == "me" and tr["source"] == "button"
    fix = next(m for m in out if m["type"] == "transcript_fixed")
    assert fix == {"type": "transcript_fixed", "index": 0, "speaker": "them"}
    assert any(m["type"] == "cues" and m.get("replies") for m in out)  # correcting the last line to Them asks for cues


def test_no_suggestion_on_backchannel(store, monkeypatch):
    S, _ = _session_fakes(monkeypatch)

    class Yeah:
        def transcribe(self, audio_, context=""):
            return "yeah", 0.95

    async def main():
        out = []

        async def send(m):
            out.append(m)

        profile = calibrated(f0=120).model.to_dict()
        s = S.DeviceSession(send, store, None, Yeah(), {"voice_profile": profile})
        await s.on_audio(np.concatenate([silence(.4), voice(1.2, .3, 230), silence(2.0)]).tobytes())
        await asyncio.sleep(0.3)
        await s.close()
        return out

    out = asyncio.run(main())
    assert any(m.get("state") == "your_turn" for m in out)
    assert not any(m["type"] == "cues" for m in out)


def test_suggestion_waits_until_they_pause(store, monkeypatch):
    S, FakeSTT = _session_fakes(monkeypatch)

    async def main():
        out = []

        async def send(m):
            out.append(m)

        profile = calibrated(f0=120).model.to_dict()
        s = S.DeviceSession(send, store, None, FakeSTT(), {"voice_profile": profile})
        # 0.9 s of silence ends the line but is short of "your turn" (1.2 s).
        await s.on_audio(np.concatenate([silence(.4), voice(1.2, .3, 230), silence(0.9)]).tobytes())
        await asyncio.sleep(0.3)
        early = list(out)
        await s.on_audio(silence(0.5).tobytes())
        await asyncio.sleep(0.3)
        await s.close()
        return early, out

    early, out = asyncio.run(main())
    assert not any(m["type"] == "cues" for m in early)
    assert any(m["type"] == "cues" and m.get("replies") for m in out)


def _scripted_session(store, monkeypatch, lines):
    """Session whose transcriber returns `lines` in order. Pitch 120 is the wearer."""
    S, _ = _session_fakes(monkeypatch)

    class Script:
        def __init__(self):
            self.lines = list(lines)

        def transcribe(self, audio_, context=""):
            return self.lines.pop(0), 0.9

    async def run(them_then_me):
        out = []

        async def send(m):
            out.append(m)

        profile = calibrated(f0=120).model.to_dict()
        s = S.DeviceSession(send, store, None, Script(), {"voice_profile": profile})
        for speaker, seconds in them_then_me:
            pitch = 230 if speaker == "them" else 120
            await s.on_audio(np.concatenate([silence(.3), voice(seconds, .3, pitch), silence(1.5)]).tobytes())
            await asyncio.sleep(0.3)
        await s.close()
        return out

    return run


def test_reply_waits_five_seconds_of_silence(store, monkeypatch):
    """A finished line is not enough. The reply waits out five seconds of quiet."""
    from server import session as S

    calls = []

    class One:
        def transcribe(self, audio_, context=""):
            return "How are you doing today?", 0.9

    async def fake_cues(person, turns, settings, moment=""):
        calls.append(moment)
        return {"replies": ["I'm good, thanks. How are you?"], "social": None, "emotion": None}

    monkeypatch.setattr(S.llm, "live_cues", fake_cues)
    monkeypatch.setattr(S.config, "CUE_MIN_INTERVAL_S", 0)
    monkeypatch.setattr(S.config, "CUE_REPLY_AFTER_S", 5)
    monkeypatch.setattr(S.config, "CUE_ICEBREAKER_AFTER_S", 3600)

    async def main():
        out = []

        async def send(m):
            out.append(m)

        profile = calibrated(f0=120).model.to_dict()
        s = S.DeviceSession(send, store, None, One(), {"voice_profile": profile})
        await s.on_audio(np.concatenate([silence(.4), voice(1.2, .3, 230), silence(4.0)]).tobytes())
        await asyncio.sleep(0.3)
        early = list(out)
        await s.on_audio(silence(1.5).tobytes())
        await asyncio.sleep(0.4)
        await s.close()
        return early, out

    early, out = asyncio.run(main())
    assert not any(m["type"] == "cues" and m.get("replies") for m in early)
    assert calls == ["They asked a question and then paused. A short answer is appropriate."]
    assert any(m["type"] == "cues" and m.get("replies") for m in out)


def test_a_few_hesitations_open_a_reply_before_five_seconds(store, monkeypatch):
    """One umm is not enough. A second stall sound asks for a reply without waiting out the quiet."""
    from server import session as S

    calls = []

    class Script:
        def __init__(self):
            self.lines = ["How are you doing today?", "umm", "ah"]

        def transcribe(self, audio_, context=""):
            return self.lines.pop(0), 0.9

    async def fake_cues(person, turns, settings, moment=""):
        calls.append([t["text"] for t in turns])
        return {"replies": ["I'm good, thanks. How are you?"], "social": None, "emotion": None}

    monkeypatch.setattr(S.llm, "live_cues", fake_cues)
    monkeypatch.setattr(S.config, "CUE_MIN_INTERVAL_S", 0)
    monkeypatch.setattr(S.config, "CUE_REPLY_AFTER_S", 5)
    monkeypatch.setattr(S.config, "CUE_ICEBREAKER_AFTER_S", 3600)

    async def main():
        out = []

        async def send(m):
            out.append(m)

        profile = calibrated(f0=120).model.to_dict()
        s = S.DeviceSession(send, store, None, Script(), {"voice_profile": profile})
        await s.on_audio(np.concatenate([silence(.4), voice(1.2, .3, 230), silence(1.2)]).tobytes())
        await asyncio.sleep(0.3)
        await s.on_audio(np.concatenate([silence(.2), voice(0.5, .3, 120), silence(0.9)]).tobytes())
        await asyncio.sleep(0.3)
        after_one = len(calls)
        await s.on_audio(np.concatenate([silence(.2), voice(0.5, .3, 120), silence(0.9)]).tobytes())
        await asyncio.sleep(0.4)
        await s.close()
        return after_one, out

    after_one, out = asyncio.run(main())
    assert after_one == 0
    assert calls and calls[0][0] == "How are you doing today?"
    assert any(m["type"] == "cues" and m.get("replies") for m in out)
    umms = [m for m in out if m["type"] == "transcript" and m.get("hesitation")]
    assert [m["text"] for m in umms] == ["umm", "ah"]


def test_an_umm_does_not_restart_the_five_seconds(store, monkeypatch):
    """Hesitating in the middle of the quiet still counts toward the five seconds."""
    from server import session as S

    calls = []

    class Script:
        def __init__(self):
            self.lines = ["How are you doing today?", "umm"]

        def transcribe(self, audio_, context=""):
            return self.lines.pop(0), 0.9

    async def fake_cues(person, turns, settings, moment=""):
        calls.append(1)
        return {"replies": ["I'm good, thanks"], "social": None, "emotion": None}

    monkeypatch.setattr(S.llm, "live_cues", fake_cues)
    monkeypatch.setattr(S.config, "CUE_MIN_INTERVAL_S", 0)
    monkeypatch.setattr(S.config, "CUE_REPLY_AFTER_S", 5)
    monkeypatch.setattr(S.config, "CUE_ICEBREAKER_AFTER_S", 3600)

    async def main():
        out = []

        async def send(m):
            out.append(m)

        profile = calibrated(f0=120).model.to_dict()
        s = S.DeviceSession(send, store, None, Script(), {"voice_profile": profile})
        await s.on_audio(np.concatenate([silence(.4), voice(1.2, .3, 230), silence(2.0)]).tobytes())
        await asyncio.sleep(0.3)
        await s.on_audio(np.concatenate([silence(.2), voice(0.5, .3, 120), silence(0.9)]).tobytes())
        await asyncio.sleep(0.3)
        mid = len(calls)
        # If the umm had zeroed the clock, this quiet would still be under five seconds.
        await s.on_audio(silence(3.0).tobytes())
        await asyncio.sleep(0.4)
        await s.close()
        return mid, out

    mid, out = asyncio.run(main())
    assert mid == 0
    assert calls
    assert any(m["type"] == "cues" and m.get("replies") for m in out)


def test_icebreaker_after_eight_seconds_of_stall(store, monkeypatch):
    """One reply at 5s stays up. At 8s one ice breaker is added under it."""
    from server import session as S

    asked = []

    class One:
        def transcribe(self, audio_, context=""):
            return "How are you doing today?", 0.9

    async def fake_cues(person, turns, settings, moment=""):
        asked.append(settings.get("num_replies"))
        return {"replies": ["I'm good, thanks. How are you?", "What about you?"],
                "social": None, "emotion": None}

    async def fake_ice(person, turns, settings):
        return {"replies": ["How's your week been?"], "social": "Try a new topic",
                "emotion": None, "kind": "icebreaker"}

    monkeypatch.setattr(S.llm, "live_cues", fake_cues)
    monkeypatch.setattr(S.llm, "icebreaker", fake_ice)
    monkeypatch.setattr(S.config, "CUE_MIN_INTERVAL_S", 0)
    monkeypatch.setattr(S.config, "CUE_REPLY_AFTER_S", 5)
    monkeypatch.setattr(S.config, "CUE_ICEBREAKER_AFTER_S", 8)

    async def main():
        out = []

        async def send(m):
            out.append(m)

        profile = calibrated(f0=120).model.to_dict()
        s = S.DeviceSession(send, store, None, One(), {"voice_profile": profile})
        await s.on_audio(np.concatenate([silence(.4), voice(1.2, .3, 230), silence(5.3)]).tobytes())
        await asyncio.sleep(0.4)
        midway = [m for m in out if m["type"] == "cues" and m.get("replies")]
        await s.on_audio(silence(3.2).tobytes())
        await asyncio.sleep(0.4)
        await s.close()
        return midway, out

    midway, out = asyncio.run(main())
    assert asked == [1]
    assert midway and midway[0]["replies"] == ["I'm good, thanks. How are you?"]
    ice = [m for m in out if m["type"] == "cues" and m.get("kind") == "icebreaker"]
    assert ice and ice[-1]["replies"] == [
        "I'm good, thanks. How are you?", "How's your week been?"]
    assert ice[-1]["social"] == "Try a new topic"
    assert out.index(ice[-1]) > out.index(midway[0])


def test_answering_cancels_the_icebreaker(store, monkeypatch):
    from server import session as S

    class Script:
        def __init__(self):
            self.lines = ["How are you doing today?", "I am doing pretty well"]

        def transcribe(self, audio_, context=""):
            return self.lines.pop(0), 0.9

    async def fake_cues(person, turns, settings, moment=""):
        return {"replies": ["I'm good, thanks"], "social": None, "emotion": None}

    async def fake_ice(person, turns, settings):
        raise AssertionError("ice breaker after the wearer already answered")

    monkeypatch.setattr(S.llm, "live_cues", fake_cues)
    monkeypatch.setattr(S.llm, "icebreaker", fake_ice)
    monkeypatch.setattr(S.config, "CUE_MIN_INTERVAL_S", 0)
    monkeypatch.setattr(S.config, "CUE_REPLY_AFTER_S", 5)
    monkeypatch.setattr(S.config, "CUE_ICEBREAKER_AFTER_S", 8)

    async def main():
        out = []

        async def send(m):
            out.append(m)

        profile = calibrated(f0=120).model.to_dict()
        s = S.DeviceSession(send, store, None, Script(), {"voice_profile": profile})
        await s.on_audio(np.concatenate([silence(.4), voice(1.2, .3, 230), silence(5.3)]).tobytes())
        await asyncio.sleep(0.4)
        await s.on_audio(np.concatenate([silence(.2), voice(1.0, .3, 120), silence(2.0)]).tobytes())
        await asyncio.sleep(0.4)
        await s.close()
        return out

    out = asyncio.run(main())
    assert any(m["type"] == "cues" and m.get("replies") for m in out)
    assert any(m["type"] == "cues" and m.get("dismissed") for m in out)
    assert not any(m.get("kind") == "icebreaker" for m in out)


def test_suggestion_clears_after_wearer_answers(store, monkeypatch):
    run = _scripted_session(store, monkeypatch, [
        "Hi, I'm Ravi, I just moved here.",
        "Welcome, where did you move from?",
    ])
    out = asyncio.run(run([("them", 1.2), ("me", 1.2)]))
    cues = [m for m in out if m["type"] == "cues"]
    assert any(m.get("replies") for m in cues)
    shown = next(i for i, m in enumerate(out) if m["type"] == "cues" and m.get("replies"))
    dismissed = next(i for i, m in enumerate(out) if m["type"] == "cues" and m.get("dismissed"))
    assert dismissed > shown
    assert not any(m["type"] == "cues" and m.get("replies") and i > dismissed for i, m in enumerate(out))


def test_suggestion_clears_when_the_wearer_answers_in_their_own_words(store, monkeypatch):
    """A real answer that is not the offered line still takes the lines down."""
    run = _scripted_session(store, monkeypatch, [
        "How are you doing today?",
        "I am doing pretty well",
        "What did you do yesterday?",
    ])
    out = asyncio.run(run([("them", 1.2), ("me", 1.2), ("them", 1.2)]))
    shown = next(i for i, m in enumerate(out) if m["type"] == "cues" and m.get("replies"))
    dismissed = next(i for i, m in enumerate(out) if m["type"] == "cues" and m.get("dismissed"))
    assert dismissed > shown
    again = [i for i, m in enumerate(out) if m["type"] == "cues" and m.get("replies") and i > dismissed]
    assert again
    me = next(m for m in out if m["type"] == "transcript" and m["text"] == "I am doing pretty well")
    assert me["speaker"] == "me"
    assert not me.get("hesitation")


def test_suggestion_stays_through_a_backchannel(store, monkeypatch):
    run = _scripted_session(store, monkeypatch, [
        "Hi, I'm Ravi, I just moved here.",
        "yeah",
    ])
    out = asyncio.run(run([("them", 1.2), ("me", 0.8)]))
    assert any(m["type"] == "cues" and m.get("replies") for m in out)
    assert not any(m["type"] == "cues" and m.get("dismissed") for m in out)


def test_suggestion_survives_noise_while_it_is_written(store, monkeypatch):
    """The dots mean a line is coming. A short noise must not wipe it."""
    from server import session as S

    started = asyncio.Event()
    release = asyncio.Event()

    class One:
        def transcribe(self, audio_, context=""):
            return "I just moved here last week", 0.9

    async def fake_cues(person, turns, settings, moment=""):
        started.set()
        await release.wait()
        return {"replies": ["Tell me more about that"], "social": None, "emotion": None}

    monkeypatch.setattr(S.llm, "live_cues", fake_cues)
    monkeypatch.setattr(S.config, "CUE_MIN_INTERVAL_S", 0)
    monkeypatch.setattr(S.config, "CUE_REPLY_AFTER_S", 1.2)
    monkeypatch.setattr(S.config, "CUE_ICEBREAKER_AFTER_S", 3600.0)

    async def main():
        out = []

        async def send(m):
            out.append(m)

        profile = calibrated(f0=120).model.to_dict()
        s = S.DeviceSession(send, store, None, One(), {"voice_profile": profile})
        await s.on_audio(np.concatenate([silence(.4), voice(1.2, .3, 230), silence(1.5)]).tobytes())
        await asyncio.wait_for(started.wait(), timeout=2)
        await s.on_audio(np.concatenate([voice(0.25, .3, 230), silence(0.9)]).tobytes())
        release.set()
        await asyncio.sleep(0.4)
        await s.close()
        return out

    out = asyncio.run(main())
    assert any(m["type"] == "cues" and m.get("pending") for m in out)
    assert any(m["type"] == "cues" and m.get("replies") == ["Tell me more about that"] for m in out)


def test_backchannel_while_suggestion_is_written_still_shows_it(store, monkeypatch):
    from server import session as S

    started = asyncio.Event()
    release = asyncio.Event()

    class Script:
        def __init__(self):
            self.lines = ["I just moved here last week", "yeah"]

        def transcribe(self, audio_, context=""):
            return self.lines.pop(0), 0.9

    async def fake_cues(person, turns, settings, moment=""):
        started.set()
        await release.wait()
        return {"replies": ["Tell me more about that"], "social": None, "emotion": None}

    monkeypatch.setattr(S.llm, "live_cues", fake_cues)
    monkeypatch.setattr(S.config, "CUE_MIN_INTERVAL_S", 0)
    monkeypatch.setattr(S.config, "CUE_REPLY_AFTER_S", 1.2)
    monkeypatch.setattr(S.config, "CUE_ICEBREAKER_AFTER_S", 3600.0)

    async def main():
        out = []

        async def send(m):
            out.append(m)

        profile = calibrated(f0=120).model.to_dict()
        s = S.DeviceSession(send, store, None, Script(), {"voice_profile": profile})
        await s.on_audio(np.concatenate([silence(.4), voice(1.2, .3, 230), silence(1.5)]).tobytes())
        await asyncio.wait_for(started.wait(), timeout=2)
        await s.on_audio(np.concatenate([silence(.2), voice(0.8, .3, 120), silence(1.2)]).tobytes())
        await asyncio.sleep(0.3)
        release.set()
        await asyncio.sleep(0.4)
        await s.close()
        return out

    out = asyncio.run(main())
    assert any(m["type"] == "cues" and m.get("replies") == ["Tell me more about that"] for m in out)
    assert not any(m["type"] == "cues" and m.get("dismissed") for m in out)


def test_suggestion_stays_through_a_cough(store, monkeypatch):
    run = _scripted_session(store, monkeypatch, [
        "Hi, I'm Ravi, I just moved here.",
        "ahem",
    ])
    out = asyncio.run(run([("them", 1.2), ("them", 0.6)]))
    assert any(m["type"] == "cues" and m.get("replies") for m in out)
    assert not any(m["type"] == "cues" and m.get("dismissed") for m in out)


def test_suggestion_stays_when_they_keep_the_subject(store, monkeypatch):
    run = _scripted_session(store, monkeypatch, [
        "Hi, I'm Ravi, I just moved here.",
        "He helped me carry the boxes inside",
    ])
    out = asyncio.run(run([("them", 1.2), ("them", 1.4)]))
    assert any(m["type"] == "cues" and m.get("replies") for m in out)
    assert not any(m["type"] == "cues" and m.get("dismissed") for m in out)


def test_cough_while_more_speech_is_transcribing_still_suggests(store, monkeypatch):
    """The real line must not be dropped just because a cough is still in the queue."""
    import threading
    from server import session as S

    class Gate:
        def __init__(self):
            self.lines = ["I just moved here last week", "ahem"]
            self.n = 0
            self.first_started = threading.Event()
            self.allow = threading.Event()

        def transcribe(self, audio_, context=""):
            i = self.n
            self.n += 1
            if i == 0:
                self.first_started.set()
                assert self.allow.wait(3), "first transcription was not released"
            return self.lines[i], 0.9

    async def fake_cues(person, turns, settings, moment=""):
        return {"replies": ["Tell me more about that"], "social": None, "emotion": None}

    monkeypatch.setattr(S.llm, "live_cues", fake_cues)
    monkeypatch.setattr(S.config, "CUE_MIN_INTERVAL_S", 0)
    monkeypatch.setattr(S.config, "CUE_REPLY_AFTER_S", 1.2)
    monkeypatch.setattr(S.config, "CUE_ICEBREAKER_AFTER_S", 3600.0)

    async def main():
        out = []

        async def send(m):
            out.append(m)

        gate = Gate()
        profile = calibrated(f0=120).model.to_dict()
        s = S.DeviceSession(send, store, None, gate, {"voice_profile": profile})
        loop = asyncio.get_running_loop()
        first = asyncio.create_task(
            s.on_audio(np.concatenate([silence(.4), voice(1.2, .3, 230), silence(1.5)]).tobytes()))
        await loop.run_in_executor(None, lambda: gate.first_started.wait(2))
        assert gate.first_started.is_set()
        await s.on_audio(np.concatenate([silence(.2), voice(0.8, .3, 230), silence(1.5)]).tobytes())
        assert s._stt_q.qsize() >= 1
        gate.allow.set()
        await first
        await asyncio.sleep(0.6)
        await s.close()
        return out

    out = asyncio.run(main())
    assert any(m["type"] == "cues" and m.get("replies") for m in out)
    assert not any(m["type"] == "cues" and m.get("dismissed") for m in out)


def test_short_uh_counts_toward_a_few_hesitations(store, monkeypatch):
    """A quick "uh" is shorter than a real turn. Two of them still ask for a reply."""
    from server import session as S

    calls = []

    class Script:
        def __init__(self):
            self.lines = ["How are you doing today?", "uh", "ah"]

        def transcribe(self, audio_, context=""):
            return self.lines.pop(0), 0.9

    async def fake_cues(person, turns, settings, moment=""):
        calls.append([t["text"] for t in turns])
        return {"replies": ["I'm good, thanks. How are you?"], "social": None, "emotion": None}

    monkeypatch.setattr(S.llm, "live_cues", fake_cues)
    monkeypatch.setattr(S.config, "CUE_MIN_INTERVAL_S", 0)
    monkeypatch.setattr(S.config, "CUE_REPLY_AFTER_S", 5)
    monkeypatch.setattr(S.config, "CUE_ICEBREAKER_AFTER_S", 3600)

    async def main():
        out = []

        async def send(m):
            out.append(m)

        profile = calibrated(f0=120).model.to_dict()
        s = S.DeviceSession(send, store, None, Script(), {"voice_profile": profile})
        await s.on_audio(np.concatenate([silence(.4), voice(1.2, .3, 230), silence(1.2)]).tobytes())
        await asyncio.sleep(0.3)
        await s.on_audio(np.concatenate([silence(.2), voice(0.25, .3, 120), silence(0.9)]).tobytes())
        await asyncio.sleep(0.4)
        after_one = len(calls)
        await s.on_audio(np.concatenate([silence(.2), voice(0.25, .3, 120), silence(0.9)]).tobytes())
        await asyncio.sleep(0.4)
        await s.close()
        return after_one, out

    after_one, out = asyncio.run(main())
    assert after_one == 0
    assert calls and calls[0][0] == "How are you doing today?"
    assert any(m["type"] == "cues" and m.get("replies") for m in out)


def test_hesitation_accepts_wearer_pitch_when_the_label_is_wrong(store):
    from server import session as S

    async def main():
        async def send(m):
            pass

        profile = calibrated(f0=120).model.to_dict()
        s = S.DeviceSession(send, store, None, None, {"voice_profile": profile, "on_hesitation": True})
        try:
            assert s._is_wearer_hesitation({"speaker": "them", "f0": 122, "source": "auto"}, "uhm")
            assert not s._is_wearer_hesitation({"speaker": "them", "f0": 230, "source": "auto"}, "uhm")
            assert s._is_wearer_hesitation({"speaker": "me", "f0": None, "source": "auto"}, "ah")
            assert not s._is_wearer_hesitation({"speaker": "me", "f0": None, "source": "auto"}, "yeah")
        finally:
            await s.close()

    asyncio.run(main())


def test_suggestion_clears_when_they_move_on(store, monkeypatch):
    run = _scripted_session(store, monkeypatch, [
        "Hi, I'm Ravi, I just moved here.",
        "I start the new job on monday actually.",
    ])
    out = asyncio.run(run([("them", 1.2), ("them", 1.4)]))
    cues = [(i, m) for i, m in enumerate(out) if m["type"] == "cues"]
    shown = [i for i, m in cues if m.get("replies")]
    dismissed = [i for i, m in cues if m.get("dismissed")]
    assert shown and dismissed and dismissed[0] > shown[0]
    # A fresh prompt may follow the new line. It must come after the old one is gone.
    later = [i for i in shown if i > dismissed[0]]
    assert not later or later[0] > dismissed[0]


# ------------------------------------------------------------ API speech-to-text
def test_replayed_line_is_not_a_second_speaker():
    from server.stt import echoes_recent_line, replayed_as_other_speaker

    said = "I just moved here last week."
    assert echoes_recent_line("I just moved here last week.", [said])
    assert echoes_recent_line("i just moved here last week", [said])
    assert echoes_recent_line("I just moved here last week. I just moved here last week.", [said])
    assert not echoes_recent_line("I just moved here last week, from Delhi.", [said])
    assert not echoes_recent_line("yeah", [said])
    assert not echoes_recent_line("How are you doing today?", [said])

    turns = [{"speaker": "me", "text": said}]
    assert replayed_as_other_speaker(said, "them", turns)
    assert not replayed_as_other_speaker(said, "me", turns)
    assert not replayed_as_other_speaker("Where did you move from?", "them", turns)


def test_transcript_does_not_repeat_the_wearer_as_them(store, monkeypatch):
    run = _scripted_session(store, monkeypatch, [
        "I just moved here last week.",
        "I just moved here last week.",
    ])
    out = asyncio.run(run([("me", 1.2), ("them", 1.2)]))
    lines = [m for m in out if m["type"] == "transcript"]
    assert [(m["speaker"], m["text"]) for m in lines] == [("me", "I just moved here last week.")]


def test_wearer_can_repeat_their_own_line(store, monkeypatch):
    run = _scripted_session(store, monkeypatch, [
        "I just moved here last week.",
        "I just moved here last week.",
    ])
    out = asyncio.run(run([("me", 1.2), ("me", 1.2)]))
    lines = [m for m in out if m["type"] == "transcript"]
    assert [m["speaker"] for m in lines] == ["me", "me"]
    assert all(m["text"] == "I just moved here last week." for m in lines)


def test_stt_keeps_a_hesitation_scored_as_non_speech():
    from server.stt import filter_segments

    text, conf = filter_segments([
        {"text": " uh", "avg_logprob": -1.4, "no_speech_prob": 0.95, "compression_ratio": 1.0},
    ])
    assert text == "uh" and conf >= 0.55
    dropped, _ = filter_segments([
        {"text": " thanks for watching", "avg_logprob": -0.2, "no_speech_prob": 0.9, "compression_ratio": 1.0},
    ])
    assert dropped == ""


def test_api_transcriber_filters_and_sends_context(monkeypatch):
    import httpx
    from server import stt
    seen = {}

    def handler(req: httpx.Request):
        seen["auth"] = req.headers["authorization"]
        seen["body"] = req.content
        return httpx.Response(200, json={"text": "...", "segments": [
            {"text": " Hi, I'm Ravi.", "avg_logprob": -0.2, "no_speech_prob": 0.01, "compression_ratio": 1.1},
            {"text": " Thanks for watching", "avg_logprob": -0.3, "no_speech_prob": 0.1, "compression_ratio": 1.0},
            {"text": " mumble mumble", "avg_logprob": -1.6, "no_speech_prob": 0.2, "compression_ratio": 1.0},
        ]})

    monkeypatch.setattr(stt.config, "STT_API_KEY", "test-key")
    t = stt.ApiTranscriber()
    t.http = httpx.Client(base_url="https://x/v1", transport=httpx.MockTransport(handler),
                          headers={"Authorization": "Bearer test-key"})
    text, conf = t.transcribe(voice(1.0, 0.3).astype(np.float32) / 32768, context="We met at the bus stop")
    assert text == "Hi, I'm Ravi." and conf > 0.7
    assert seen["auth"] == "Bearer test-key"
    assert b"We met at the bus stop" in seen["body"] and b"RIFF" in seen["body"]


def test_missing_key_gives_clear_error(monkeypatch):
    from server import stt
    monkeypatch.setattr(stt.config, "STT_API_KEY", "")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        stt.ApiTranscriber()


# ------------------------------------------------------------ model resolution / errors
def _mock_llm(monkeypatch, handler):
    import httpx
    monkeypatch.setattr(llm, "_client", httpx.AsyncClient(base_url="https://x/v1",
                                                          transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(llm, "_sem", asyncio.Semaphore(4))


def test_retired_model_falls_back(monkeypatch):
    import httpx
    _mock_llm(monkeypatch, lambda req: httpx.Response(200, json={"data": [
        {"id": "openai/gpt-oss-120b"}, {"id": "whisper-large-v3-turbo"}]}))
    monkeypatch.setattr(config, "LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setattr(config, "LLM_SUMMARY_MODEL", "")
    monkeypatch.setattr(config, "LLM_CHECK_MODEL", "retired-check")
    monkeypatch.setattr(config, "STT_API_MODEL", "whisper-large-v3")
    monkeypatch.setattr(config, "STT_API_BASE_URL", config.LLM_BASE_URL)
    asyncio.run(llm.resolve_models())
    assert config.LLM_MODEL == "openai/gpt-oss-120b"
    assert config.LLM_CHECK_MODEL == "openai/gpt-oss-120b"
    assert config.STT_API_MODEL == "whisper-large-v3-turbo"


def test_reasoning_model_request_and_404_message(monkeypatch):
    import httpx
    sent = {}

    def handler(req):
        body = json.loads(req.content)
        sent.update(body)
        if body["model"] == "gone-model":
            return httpx.Response(404, json={"error": {"message": "The model `gone-model` has been decommissioned"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"replies": ["Hi"]}'}}]})

    _mock_llm(monkeypatch, handler)
    out = asyncio.run(llm.chat_json("s", "u", model="openai/gpt-oss-120b", max_tokens=180))
    assert out == {"replies": ["Hi"]}
    assert sent["reasoning_effort"] == "low" and sent["max_tokens"] > 180
    with pytest.raises(llm.LLMError, match="decommissioned"):
        asyncio.run(llm.chat_json("s", "u", model="gone-model"))


# ------------------------------------------------------------ background noise and echoed lines
def hiss(seconds, std):
    """Steady white noise: an AC, a laptop fan, mic hiss."""
    return np.clip(rng.normal(0, std, int(seconds * 16000)), -32767, 32767).astype(np.int16)


def fan(seconds, std, stream=None):
    """Low, steady rumble: a ceiling fan or traffic."""
    stream = rng if stream is None else stream
    x = np.convolve(stream.normal(0, 1, int(seconds * 16000)), np.ones(40) / 40, mode="same")
    return np.clip(x / x.std() * std, -32767, 32767).astype(np.int16)


def test_steady_noise_is_never_a_line():
    """Nobody is talking: fan or hiss must not become a transcript, at any loudness."""
    for noise in (hiss(6, 300), hiss(6, 1000), fan(6, 300), fan(6, 1500)):
        det = calibrated(f0=120)
        before = json.dumps(det.model.to_dict())
        ev = run(det, silence(0.5), noise, silence(1.5))
        assert not utterances(ev)
        assert any(e["type"] == "noise" for e in ev)
        assert json.dumps(det.model.to_dict()) == before     # noise does not teach the voice model


def test_partner_still_heard_over_a_steady_fan():
    det = calibrated(f0=120)
    room = fan(9, 300)
    talk = np.zeros_like(room)
    line = voice(1.5, 0.15, 230)
    talk[72000:72000 + len(line)] = line                      # they speak 4.5 s in
    mixed = np.clip(room.astype(np.int32) + talk, -32767, 32767).astype(np.int16)
    u = utterances(run(det, silence(0.5), mixed, silence(1.5)))
    assert [e["speaker"] for e in u] == ["them"]
    assert u[0]["snr"] >= config.SPEECH_MIN_SNR_DB


def test_quiet_partner_in_a_quiet_room_still_heard():
    det = calibrated(f0=120, amp=0.4)
    u = utterances(run(det, silence(0.5), voice(1.5, 0.03, 230), silence(1.5)))
    assert len(u) == 1 and u[0]["speaker"] == "them" and u[0]["snr"] > 0


def test_noise_does_not_close_an_open_floor():
    det = calibrated(f0=120)
    run(det, silence(0.4), voice(1.2, 0.3, 230), silence(1.6))
    assert det.floor_open
    ev = run(det, fan(2.0, 1000), silence(0.9))
    assert not utterances(ev) and det.floor_open


def test_weak_echo_catches_short_and_near_copies():
    import time
    from server.stt import weak_echo

    now = time.time()
    hello = [{"speaker": "me", "text": "Hello.", "t": now - 1}]
    assert weak_echo("hello", hello, now)
    assert weak_echo("Hello!", hello, now)
    assert not weak_echo("Hi there", hello, now)
    assert not weak_echo("hello", [{**hello[0], "t": now - 60}], now)    # long ago: not an echo
    moved = [{"speaker": "me", "text": "I just moved here last week.", "t": now - 2}]
    assert weak_echo("just moved here last week", moved, now)          # a piece of it
    assert weak_echo("I just moved here last weekend", moved, now)     # nearly the same
    assert not weak_echo("Where did you move from?", moved, now)
    assert not weak_echo("last week?", moved, now)                     # short real reply kept


def test_weak_clip_is_quiet_or_unsure():
    from server.session import DeviceSession

    assert DeviceSession._weak_clip({"snr": 8.0}, 0.9)
    assert DeviceSession._weak_clip({"snr": 30.0}, 0.4)
    assert not DeviceSession._weak_clip({"snr": 30.0}, 0.9)


def test_too_many_words_for_the_clip():
    from server.stt import too_many_words

    long_line = "I think we should all go to the market together early tomorrow morning"
    assert too_many_words(long_line, 500)
    assert not too_many_words(long_line, 3000)
    assert not too_many_words("Hello there", 300)
    assert not too_many_words(long_line, None)


def test_faint_noise_is_not_turned_up_to_full_volume():
    from server.stt import _preprocess

    faint = (rng.normal(0, 1e-4, 16000)).astype(np.float32)
    assert np.max(np.abs(_preprocess(faint))) < 0.05
    speech = voice(1.0, 0.3).astype(np.float32) / 32768
    assert abs(np.max(np.abs(_preprocess(speech))) - 0.9) < 1e-3


def test_whisper_noise_phrases_dropped_real_thanks_kept():
    from server.stt import filter_segments

    def seg(text):
        return [{"text": text, "avg_logprob": -0.3, "no_speech_prob": 0.1, "compression_ratio": 1.0}]

    for junk in (" Thank you so much for watching!", " Thank you very much.", " Subtitles by the Amara.org community",
                 " Please like and subscribe."):
        assert filter_segments(seg(junk))[0] == "", junk
    assert filter_segments(seg(" Thanks."))[0] == "Thanks."
    assert filter_segments(seg(" Hello."))[0] == "Hello."


def _clip_session(store, monkeypatch, script):
    """script: (speaker, seconds, amplitude, text, confidence). Pitch 120 is the wearer."""
    S, _ = _session_fakes(monkeypatch)
    seen = {"contexts": [], "calls": 0}

    class Script:
        def __init__(self):
            self.lines = [(text, conf) for *_, text, conf in script]

        def transcribe(self, audio_, context=""):
            seen["calls"] += 1
            seen["contexts"].append(context)
            return self.lines.pop(0)

    async def main():
        out = []

        async def send(m):
            out.append(m)

        profile = calibrated(f0=120).model.to_dict()
        s = S.DeviceSession(send, store, None, Script(), {"voice_profile": profile})
        for speaker, seconds, amp, *_ in script:
            pitch = 230 if speaker == "them" else 120
            await s.on_audio(np.concatenate([silence(.3), voice(seconds, amp, pitch), silence(1.5)]).tobytes())
            await asyncio.sleep(0.3)
        await s.close()
        return out

    out = asyncio.run(main())
    return [(m["speaker"], m["text"]) for m in out if m["type"] == "transcript"], seen


def test_my_hello_is_not_repeated_as_them(store, monkeypatch):
    """The reported bug: I say hello, a faint sound right after comes back as 'Them: hello'."""
    lines, seen = _clip_session(store, monkeypatch, [
        ("me", 0.8, 0.3, "Hello.", 0.9),
        ("them", 0.6, 0.3, "Hello.", 0.45),       # Whisper unsure: an echo, not a reply
    ])
    assert lines == [("me", "Hello.")]
    assert seen["contexts"] == ["", ""]           # the last lines are not fed back to Whisper


def test_a_clear_hello_back_is_kept(store, monkeypatch):
    lines, _ = _clip_session(store, monkeypatch, [
        ("me", 0.8, 0.3, "Hello.", 0.9),
        ("them", 0.8, 0.3, "Hello!", 0.9),
    ])
    assert lines == [("me", "Hello."), ("them", "Hello!")]


def test_context_prompt_can_be_turned_back_on(store, monkeypatch):
    monkeypatch.setattr(config, "STT_CONTEXT_PROMPT", True)
    _, seen = _clip_session(store, monkeypatch, [
        ("me", 0.8, 0.3, "Hello there.", 0.9),
        ("them", 1.0, 0.3, "Nice to see you.", 0.9),
    ])
    assert seen["contexts"] == ["", "Hello there."]


def test_sentence_invented_from_a_blip_is_dropped(store, monkeypatch):
    lines, _ = _clip_session(store, monkeypatch, [
        ("them", 0.5, 0.3, "I think we should all go to the market together early tomorrow morning", 0.9),
    ])
    assert lines == []


def test_fan_noise_never_reaches_speech_to_text(store, monkeypatch):
    S, _ = _session_fakes(monkeypatch)
    calls = []

    class Counting:
        def transcribe(self, audio_, context=""):
            calls.append(1)
            return "Thank you.", 0.9

    async def main():
        out = []

        async def send(m):
            out.append(m)

        profile = calibrated(f0=120).model.to_dict()
        s = S.DeviceSession(send, store, None, Counting(), {"voice_profile": profile})
        for noise in (fan(5, 1500), hiss(5, 300)):
            await s.on_audio(np.concatenate([silence(.5), noise, silence(1.5)]).tobytes())
            await asyncio.sleep(0.2)
        await s.close()
        return out

    out = asyncio.run(main())
    assert calls == []
    assert not any(m["type"] == "transcript" for m in out)


def test_my_line_heard_while_a_fan_is_running():
    """The fan's random peaks used to hold the line open, merging fan and speech into one clip."""
    det = calibrated(f0=120)
    # Own stream: earlier tests also draw silence from the shared generator.
    room = fan(7, 800, np.random.default_rng(0))
    talk = np.zeros_like(room)
    line = voice(1.5, 0.3, 120)
    talk[48000:48000 + len(line)] = line                      # fan on for 3 s, then I speak
    mixed = np.clip(room.astype(np.int32) + talk, -32767, 32767).astype(np.int16)
    u = utterances(run(det, mixed))
    assert [e["speaker"] for e in u] == ["me"]
    assert u[0]["duration"] < 3.0                             # just the line, not the fan before it
