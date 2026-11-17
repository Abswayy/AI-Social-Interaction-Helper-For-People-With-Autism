"""Turns per-frame face detections into a stable "who am I talking to" answer.

* Smooths recognition over a few frames so one bad frame can't flip identity.
* Auto-registers an unknown face only after it has been seen consistently for
  a while (avoids registering passers-by and blurry half-faces).
"""
from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from . import config
from .face import Face, encode_jpeg
from .store import Store


@dataclass
class IdentityState:
    person_id: int | None = None
    similarity: float = 0.0
    just_registered: bool = False
    bbox: tuple | None = None
    face_visible: bool = False


class IdentityTracker:
    def __init__(self, store: Store, auto_register: bool = config.AUTO_REGISTER):
        self.store = store
        self.auto_register = auto_register
        self._votes: deque = deque(maxlen=5)
        self._unknown: list[np.ndarray] = []
        self._best_crop: tuple[float, np.ndarray] | None = None
        self._photo_seen: set[int] = set()
        self._unknown_since: float | None = None
        self.current: int | None = None
        self.last_face_at: float = 0.0

    def update(self, faces: list[Face], now: float | None = None) -> IdentityState:
        now = now or time.time()
        if not faces:
            self._votes.append(None)
            self._reset_unknown_if_stale(now)
            return IdentityState(person_id=self.current, face_visible=False)

        face = faces[0]                      # largest = conversation partner
        self.last_face_at = now
        pid, sim = self.store.match(face.embedding)
        self._votes.append(pid)

        if pid is not None:
            self._unknown.clear(); self._unknown_since = None
            winner, n = Counter(v for v in self._votes if v is not None).most_common(1)[0]
            if n >= 2 or self.current == winner:
                if self.current != winner:
                    self.store.touch(winner)
                self.current = winner
                # Learn new angles only from confident matches (prevents drift).
                if sim > config.FACE_MATCH_THRESHOLD + 0.08:
                    self.store.add_embedding(winner, face.embedding)
                    if face.crop is not None and not self._photo_checked(winner):
                        self.store.set_photo(winner, encode_jpeg(face.crop))
            return IdentityState(self.current, sim, False, face.bbox, True)

        # ---- unknown face ----
        if not self.auto_register:
            return IdentityState(None, sim, False, face.bbox, True)
        if self._unknown and float(np.mean([u @ face.embedding for u in self._unknown])) < 0.45:
            # Looks like a different unknown person than we were accumulating.
            self._unknown.clear(); self._unknown_since = None
        self._unknown.append(face.embedding)
        self._unknown_since = self._unknown_since or now
        if face.crop is not None and (self._best_crop is None or face.score > self._best_crop[0]):
            self._best_crop = (face.score, face.crop)

        if (len(self._unknown) >= config.AUTO_REGISTER_FRAMES
                and now - self._unknown_since >= config.AUTO_REGISTER_MIN_SECONDS):
            new_id = self.store.create_person(source="auto")
            # Store a few diverse samples, best-quality-first is not knowable, so spread out.
            for emb in self._unknown[:: max(1, len(self._unknown) // 3)][:3]:
                self.store.add_embedding(new_id, emb, force=True)
            if self._best_crop is not None:
                self.store.set_photo(new_id, encode_jpeg(self._best_crop[1]))
            self._unknown.clear(); self._unknown_since = None; self._best_crop = None
            self._votes.clear(); self._votes.append(new_id)
            self.current = new_id
            return IdentityState(new_id, 1.0, True, face.bbox, True)

        return IdentityState(None, sim, False, face.bbox, True)

    def _photo_checked(self, pid: int) -> bool:
        """Only look up 'has photo?' once per person per session."""
        if pid in self._photo_seen:
            return True
        self._photo_seen.add(pid)
        return False

    def _reset_unknown_if_stale(self, now: float):
        if self._unknown_since and now - self.last_face_at > 3:
            self._unknown.clear(); self._unknown_since = None
