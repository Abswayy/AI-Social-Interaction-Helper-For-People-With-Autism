"""Streaming voice activity, turn-taking and who-is-speaking.

Input: 16 kHz mono int16 PCM in chunks of any size.
Output events (dicts):
  {"type": "speech_start", "speaker": "me"|"them"}
  {"type": "false_start"}   voiced blip shorter than a real turn; the pause clock is unchanged
  {"type": "noise"}         sound that is not a voice (fan, traffic, faint background talk);
                            never transcribed, never teaches the voice model, pause clock unchanged
  {"type": "utterance", "audio": float32 ndarray, "speaker", "duration", "db", "snr", "f0", "source"}
  {"type": "your_turn"}   the other person finished and paused -> wearer may speak
  {"type": "long_pause"}  silence has lasted LONG_PAUSE_MS since the last utterance
  {"type": "overlap"}     wearer started talking while the other person was mid-sentence

Who is speaking - two modes, switchable live:
  * auto   : voice pitch + loudness compared with the wearer's calibrated voice
             and a learned profile of the other speaker. Works on a laptop
             (pitch) and better on glasses (mic at the wearer's mouth = loud).
  * manual : the wearer holds a button / Space / glasses touchpad while talking.
In auto mode holding the button still wins, and every button press and every
correction (tap a transcript line) teaches the auto model.
"""
from __future__ import annotations

import collections
import math

import numpy as np
import webrtcvad

from . import config

FRAME_MS = 20
FRAME_SAMPLES = config.SAMPLE_RATE * FRAME_MS // 1000   # 320
FRAME_BYTES = FRAME_SAMPLES * 2
F0_MIN, F0_MAX = 70.0, 400.0


def frame_db(frame: np.ndarray) -> float:
    rms = math.sqrt(float(np.mean(frame.astype(np.float32) ** 2)) + 1e-9)
    return 20 * math.log10(rms / 32768 + 1e-12)


def estimate_f0(window: np.ndarray) -> float | None:
    """Pitch of a ~40 ms window by normalised autocorrelation; None if unvoiced."""
    x = window.astype(np.float32)
    x = x - x.mean()
    if float(np.dot(x, x)) < 1e3:
        return None
    n = len(x)
    spec = np.fft.rfft(x, 2 * n)
    r = np.fft.irfft(spec * np.conj(spec))[:n]
    r /= r[0] + 1e-9
    lo, hi = int(config.SAMPLE_RATE / F0_MAX), int(config.SAMPLE_RATE / F0_MIN)
    lag = lo + int(np.argmax(r[lo:hi]))
    if r[lag] < 0.45:
        return None
    return config.SAMPLE_RATE / lag


def _pitch_track(frames: list[np.ndarray], voiced: list[bool]) -> list[float | None]:
    return [estimate_f0(np.concatenate([frames[i], frames[i + 1]]))
            for i in range(len(frames) - 1) if voiced[i] and voiced[i + 1]]


def voice_stats(frames: list[np.ndarray], voiced: list[bool]) -> tuple[float | None, float]:
    """-> (median pitch or None, share of voiced windows that have a pitch at all).

    A person's voice has a pitch on most vowels. Fan hum, hiss, traffic and
    keyboard clicks do not, even when the VAD calls them speech.
    """
    track = _pitch_track(frames, voiced)
    est = [e for e in track if e]
    f0 = float(np.median(est)) if len(est) >= 5 else None
    return f0, (len(est) / len(track) if track else 0.0)


def utterance_f0(frames: list[np.ndarray], voiced: list[bool]) -> float | None:
    return voice_stats(frames, voiced)[0]


class NoiseFloor:
    """The room's background level, in dBFS.

    A low percentile of the last few seconds of frame levels. People leave gaps
    between words, so the quiet end of that window is the room, not the voice.
    A fan or traffic that starts and keeps going becomes the new floor within
    about 2.5 s, after which it no longer counts as speech.
    """
    WINDOW = 150          # 3 s of 20 ms frames
    PERCENTILE = 15
    WARMUP = 25           # first 0.5 s: quietest frame so far (a fan running at switch-on is the room)

    def __init__(self):
        self._levels: collections.deque = collections.deque(maxlen=self.WINDOW)
        self._count = 0
        self.db = config.NOISE_FLOOR_MIN_DB

    def update(self, db: float):
        self._levels.append(db)
        self._count += 1
        if len(self._levels) < self.WARMUP:
            p = min(self._levels)
        elif self._count % 5 == 0:
            p = float(np.percentile(np.fromiter(self._levels, float, len(self._levels)), self.PERCENTILE))
        else:
            return
        self.db = min(max(p, config.NOISE_FLOOR_MIN_DB), config.NOISE_FLOOR_MAX_DB)


def semitones(a: float, b: float) -> float:
    return 12 * abs(math.log2(a / b))


class SpeakerModel:
    """Two tiny voice profiles: {db, f0}. Distance in (loudness, pitch) space."""
    DB_SCALE, ST_SCALE = 8.0, 3.0

    def __init__(self, data: dict | None = None):
        data = data or {}
        self.me: dict | None = data.get("me")
        self.other: dict | None = data.get("other")

    @property
    def calibrated(self) -> bool:
        return bool(self.me)

    def to_dict(self) -> dict:
        return {"me": self.me, "other": self.other}

    def _dist(self, prof: dict, db: float, f0: float | None) -> float:
        d = ((db - prof["db"]) / self.DB_SCALE) ** 2
        if f0 and prof.get("f0"):
            d += (semitones(f0, prof["f0"]) / self.ST_SCALE) ** 2
        else:
            d *= 2            # no pitch: lean on loudness alone
        return d

    def classify(self, db: float, f0: float | None) -> tuple[str, float]:
        """-> (speaker, confidence 0..1)."""
        if not self.me:
            return "them", 0.0
        dm = self._dist(self.me, db, f0)
        if self.other and self.other.get("n", 0) >= 2:
            do = self._dist(self.other, db, f0)
            return ("me" if dm < do else "them"), abs(do - dm) / (do + dm + 1e-9)
        # No profile of the other person yet: "me" only if close to my voice.
        return ("me", max(0.0, 1 - dm / 1.5)) if dm < 1.5 else ("them", min(1.0, (dm - 1.5) / 3))

    def learn(self, speaker: str, db: float, f0: float | None, weight: float):
        if speaker == "me":
            if not self.me:
                self.me = {"db": db, "f0": f0}
                return
            target = self.me
        else:
            if not self.other:
                self.other = {"db": db, "f0": f0, "n": 1}
                return
            target = self.other
            target["n"] = target.get("n", 0) + 1
        target["db"] += weight * (db - target["db"])
        if f0:
            target["f0"] = f0 if not target.get("f0") else target["f0"] * (f0 / target["f0"]) ** weight


class TurnDetector:
    def __init__(self, voice_profile: dict | None = None, mode: str = "auto"):
        self.vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
        self.floor = NoiseFloor()
        self._start_floor = config.NOISE_FLOOR_MIN_DB   # room level when the current sound began
        self.model = SpeakerModel(voice_profile)
        self.mode = mode                 # "auto" | "manual"
        self.ptt = False                 # hold-to-talk currently pressed
        self._buf = b""
        self._ring = collections.deque(maxlen=10)          # 200 ms pre-roll + start decision
        self._in_speech = False
        self._frames: list[np.ndarray] = []
        self._voiced: list[bool] = []
        self._dbs: list[float] = []
        self._ptt_frames = 0
        self._silence_frames = 0
        self._since_last_end_ms = 0
        self._last_speaker: str | None = None
        self._your_turn_sent = True
        self._long_pause_sent = True
        self._overlap_sent = False
        # Quiet time since the last real line. Silence and stalls both count.
        # A real utterance does not zero this; the session does, once it knows
        # the words were a real continuation and not an "umm".
        self._stall_ms = 0
        self._gap_ms = 0              # silence already counted when a speech burst starts
        self.start_speaker = "them"
        self._calibrating = False
        self._calib: list[tuple[np.ndarray, bool, float]] = []

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def last_speaker(self) -> str | None:
        return self._last_speaker

    @property
    def floor_open(self) -> bool:
        """The other person has finished and paused long enough for the wearer to speak."""
        return (not self._in_speech and self._last_speaker == "them"
                and self._since_last_end_ms >= config.YOUR_TURN_MS)

    @property
    def long_pause_open(self) -> bool:
        """Silence since the last utterance has lasted long enough for the long-pause toggle."""
        return (not self._in_speech and self._last_speaker is not None
                and self._since_last_end_ms >= config.LONG_PAUSE_MS)

    @property
    def stall_ms(self) -> int:
        """Quiet, coughs, and stalls since the session last saw a real line."""
        return self._stall_ms

    def reset_stall(self):
        """A real line continued the conversation. Only the quiet after it counts."""
        self._stall_ms = 0 if self._in_speech else self._since_last_end_ms

    def restart_stall_now(self):
        """The wearer just asked for a line. Quiet time starts from this moment."""
        self._stall_ms = 0

    def add_stall_ms(self, ms: int):
        """An umm, a cough, or a backchannel is not the conversation moving on."""
        if ms > 0:
            self._stall_ms += int(ms)

    # ---------- calibration: wearer speaks for ~5 s ----------
    def start_calibration(self):
        self._calibrating, self._calib = True, []

    def finish_calibration(self) -> dict | None:
        self._calibrating = False
        voiced = [c for c in self._calib if c[1]]
        if len(voiced) < 25:              # < 0.5 s of voiced audio
            return None
        db = float(np.median([c[2] for c in voiced]))
        f0 = utterance_f0([c[0] for c in self._calib], [c[1] for c in self._calib])
        self.model.me = {"db": db, "f0": f0}
        return self.model.to_dict()

    # ---------- streaming ----------
    def feed(self, pcm: bytes) -> list[dict]:
        self._buf += pcm
        events: list[dict] = []
        while len(self._buf) >= FRAME_BYTES:
            raw, self._buf = self._buf[:FRAME_BYTES], self._buf[FRAME_BYTES:]
            events.extend(self._frame(raw))
        return events

    def _decide(self, db: float, f0: float | None, ptt_frac: float) -> tuple[str, str, float]:
        """-> (speaker, source, confidence)."""
        if self._calibrating:
            return "me", "calibration", 1.0
        if ptt_frac >= 0.3:
            return "me", "button", 1.0
        if self.mode == "manual":
            return "them", "button", 1.0
        spk, conf = self.model.classify(db, f0)
        return spk, "auto", conf

    def _frame(self, raw: bytes) -> list[dict]:
        ev: list[dict] = []
        frame = np.frombuffer(raw, dtype=np.int16)
        vad_says = self.vad.is_speech(raw, config.SAMPLE_RATE)
        db = frame_db(frame)
        room = self.floor.db
        self.floor.update(db)
        # The VAD alone lets fans, hiss and far-off chatter through. A frame is
        # speech only if it is also clearly louder than the room right now.
        # (Calibration keeps the plain VAD: the wearer is told to talk, so it is speech.)
        voiced = vad_says and (self._calibrating or db >= room + config.SPEECH_FRAME_SNR_DB)
        if self._calibrating:
            self._calib.append((frame, vad_says, db))

        if not self._in_speech:
            self._ring.append((frame, voiced, db))
            self._since_last_end_ms += FRAME_MS
            self._stall_ms += FRAME_MS
            # Start when >= 70% of the last 200 ms is voiced (rejects clicks/bumps).
            if len(self._ring) == self._ring.maxlen and sum(v for _, v, _ in self._ring) >= 7:
                self._in_speech = True
                self._start_floor = room
                self._gap_ms = self._since_last_end_ms
                self._frames = [f for f, _, _ in self._ring]
                self._voiced = [v for _, v, _ in self._ring]
                self._dbs = [d for _, v, d in self._ring if v]
                self._ptt_frames = len(self._dbs) if self.ptt else 0
                self._ring.clear()
                self._silence_frames = 0
                self._overlap_sent = False
                self.start_speaker = self._decide(float(np.median(self._dbs)),
                                                  utterance_f0(self._frames, self._voiced),
                                                  1.0 if self.ptt else 0.0)[0]
                ev.append({"type": "speech_start", "speaker": self.start_speaker})
            else:
                if (not self._your_turn_sent and self._last_speaker == "them"
                        and self._since_last_end_ms >= config.YOUR_TURN_MS):
                    self._your_turn_sent = True
                    ev.append({"type": "your_turn"})
                if (not self._long_pause_sent and self._last_speaker is not None
                        and self._since_last_end_ms >= config.LONG_PAUSE_MS):
                    self._long_pause_sent = True
                    ev.append({"type": "long_pause"})
            return ev

        # ---- inside an utterance ----
        prev = self._frames[-1]
        self._frames.append(frame)
        self._voiced.append(voiced)
        if voiced:
            self._dbs.append(db)
            self._ptt_frames += self.ptt
            # Only a voice (a pitch) holds the line open. A fan's random peaks
            # used to keep resetting the end-of-turn clock, so the line never
            # ended and the noise and the next sentence came out as one clip.
            if estimate_f0(np.concatenate([prev, frame])) is not None:
                self._silence_frames = 0
            else:
                self._silence_frames += 1
            # Wearer barging in on the partner (auto mode: my voice appears mid-utterance).
            if (self.start_speaker == "them" and not self._overlap_sent and self.mode == "auto"
                    and self.model.calibrated and len(self._dbs) > 30 and len(self._dbs) % 10 == 0):
                tail_f0 = utterance_f0(self._frames[-16:], self._voiced[-16:])
                if self.model.classify(float(np.median(self._dbs[-15:])), tail_f0)[0] == "me":
                    self._overlap_sent = True
                    ev.append({"type": "overlap"})
        else:
            self._silence_frames += 1

        dur_ms = len(self._frames) * FRAME_MS
        if self._silence_frames * FRAME_MS >= config.END_OF_TURN_MS or dur_ms >= config.MAX_UTTERANCE_S * 1000:
            ev.extend(self._close())
        return ev

    def press(self, down: bool) -> list[dict]:
        """Hold-to-talk button changed. Pressing while the partner talks = overlap."""
        self.ptt = down
        if down and self._in_speech and self.start_speaker == "them" and not self._overlap_sent:
            self._overlap_sent = True
            return [{"type": "overlap"}]
        return []

    def _close(self) -> list[dict]:
        self._in_speech = False
        frames, voiced, dbs, ptt = self._frames, self._voiced, self._dbs, self._ptt_frames
        blip_ms = len(frames) * FRAME_MS
        self._frames, self._voiced, self._dbs = [], [], []
        db = float(np.median(dbs)) if dbs else None
        f0, pitched = voice_stats(frames, voiced)
        snr = (db - self._start_floor) if db is not None else 0.0
        # Not a voice: barely above the room, or no pitch in it (fan, hiss,
        # traffic, keyboard, faint chatter at the next table). Sending this to
        # Whisper is what produced lines nobody said. It also must not teach
        # the me/them voice model. Like a cough, it leaves the pause clock alone.
        if not self._calibrating and (snr < config.SPEECH_MIN_SNR_DB
                                      or pitched < config.SPEECH_MIN_PITCHED):
            self._since_last_end_ms = self._gap_ms + blip_ms
            self._stall_ms += blip_ms
            return [{"type": "noise", "speaker": self.start_speaker,
                     "db": round(db, 1) if db is not None else None,
                     "snr": round(snr, 1), "pitched": round(pitched, 2),
                     "duration": blip_ms / 1000}]
        # A cough or a click is not a turn. Wiping the pause here used to close
        # the floor after "your turn" had already opened, and the suggestion died.
        if len(dbs) * FRAME_MS < config.MIN_UTTERANCE_MS:
            # Not a turn, so the pause clock stays. The clip is kept so a short
            # "uh" can still be read as hesitation. The stall clock keeps the blip.
            self._since_last_end_ms = self._gap_ms + blip_ms
            self._stall_ms += blip_ms
            clip = (np.concatenate(frames).astype(np.float32) / 32768.0
                    if frames else np.zeros(0, np.float32))
            return [{"type": "false_start", "speaker": self.start_speaker, "audio": clip,
                     "db": round(db, 1) if db is not None else None,
                     "snr": round(snr, 1),
                     "f0": round(f0, 1) if f0 else None}]
        self._since_last_end_ms = self._silence_frames * FRAME_MS
        speaker, source, conf = self._decide(db, f0, ptt / max(1, len(dbs)))
        # Free training data: button presses are certain; confident auto calls teach a little.
        if source == "button" and (self.model.calibrated or speaker == "me"):
            self.model.learn(speaker, db, f0, 0.3)
        elif source == "auto" and conf > 0.4:
            self.model.learn(speaker, db, f0, 0.05 if speaker == "me" else 0.2)
        self._last_speaker = speaker
        self._your_turn_sent = speaker != "them"
        self._long_pause_sent = False
        audio = np.concatenate(frames).astype(np.float32) / 32768.0
        return [{"type": "utterance", "audio": audio, "speaker": speaker, "source": source,
                 "confidence": round(conf, 2), "duration": len(audio) / config.SAMPLE_RATE,
                 "voiced_ms": len(dbs) * FRAME_MS,
                 "db": round(db, 1), "snr": round(snr, 1),
                 "f0": round(f0, 1) if f0 else None}]

    def correct(self, db: float, f0: float | None, speaker: str):
        """User tapped a transcript line to fix who said it."""
        self.model.learn(speaker, db, f0, 0.4)
