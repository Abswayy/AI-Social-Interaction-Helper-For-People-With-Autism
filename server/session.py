"""One DeviceSession per connected device (laptop browser today, glasses next week).

The session knows nothing about *how* frames and audio arrive or how captions
are drawn: it receives bytes and calls `send(dict)`. That is the whole
integration surface for new hardware.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, Callable

from . import config, llm
from .audio import SpeakerModel, TurnDetector, semitones
from .cues import (MOMENT_TEXT, cue_decision, hesitation_hits, is_ignorable, needs_repeat,
                   suggestion_expired, wearer_hesitating)
from .face import FaceEngine, decode_jpeg
from .identity import IdentityTracker
from .store import Store
from .stt import replayed_as_other_speaker, too_many_words, weak_echo

log = logging.getLogger("session")

FACE_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="face")
STT_POOL = ThreadPoolExecutor(max_workers=config.STT_WORKERS, thread_name_prefix="stt")

DEFAULT_SETTINGS = {
    "num_replies": 2,       # how many reply suggestions to show
    "social": True,         # social-cue tip
    "emotion": True,        # partner's emotional tone
    "turn_hints": True,     # "your turn" / "let them finish"
    "max_words": 10,        # caption length on the display
    "speaker_mode": "auto",  # auto = voice detection, manual = hold-to-talk only
    "voice_profile": None,   # {"me": {db, f0}, "other": {...}} learned per wearer
}


class Conversation:
    def __init__(self):
        self.started = time.time()
        self.turns: list[dict] = []
        self.person_frames: Counter = Counter()

    @property
    def person_id(self) -> int | None:
        return self.person_frames.most_common(1)[0][0] if self.person_frames else None


class DeviceSession:
    def __init__(self, send: Callable[[dict], Awaitable[None]], store: Store,
                 faces: FaceEngine | None, stt, settings: dict | None = None):
        self.send = send
        self.store = store
        self.faces = faces
        self.stt = stt
        self.settings = {**DEFAULT_SETTINGS, **(settings or {})}
        self.turns = TurnDetector(self.settings.get("voice_profile"), self.settings["speaker_mode"])
        self.identity = IdentityTracker(store)
        self.conv = Conversation()
        self.last_speech_at = 0.0
        self._face_busy = False
        self._announced: int | None = None
        self._stt_q: asyncio.Queue = asyncio.Queue()
        self._cue_task: asyncio.Task | None = None
        self._last_cue_at = 0.0
        self._cued_for: int | None = None          # transcript index we already asked about
        self._pending: tuple[int, str] | None = None  # (index, moment or "unclear") waiting for a pause
        self._placeholder = False                  # HUD is showing "…" for an in-flight suggestion
        self._showing_cues = False                 # a reply or social line is on the lens
        self._shown_replies: list[str] = []
        self._stale_after: int | None = None       # cues for this transcript index or earlier are done
        self._unclear_for: int | None = None
        self._opener_task: asyncio.Task | None = None
        self._opener_gen = 0                       # bumped to drop a starter that is no longer the one to show
        self._hesitated = False                    # wearer just sounded unsure ("umm", "ah")
        self._hearing = 0                          # utterances queued and not yet transcribed
        self._stall_on = False                     # a real line has started the quiet clock
        self._stall_gen = 0                        # bumped when the conversation actually continues
        self._cue_gen = 0                          # bumped so a cancelled reply cannot clear the next one
        self._reply_sent = False                   # this stall already asked for a direct reply
        self._ice_sent = False                     # this stall already asked for an ice breaker
        self._ice_task: asyncio.Task | None = None
        self._stall_reply: str | None = None      # the one line from the 5s mark
        self._stall_ice: str | None = None        # the one line added at the 8s mark
        self._stall_emotion: str | None = None
        self._hesitation_hits = 0                  # umms and ahhs since the last real line
        self._tasks = [asyncio.create_task(self._stt_worker()),
                       asyncio.create_task(self._watchdog())]

    # ------------------------------------------------------------ inputs
    def update_settings(self, s: dict):
        if s.get("self_voice_db") is not None and not s.get("voice_profile"):
            s["voice_profile"] = {"me": {"db": s["self_voice_db"], "f0": None}}   # older clients
        self.settings.update({k: v for k, v in s.items() if k in DEFAULT_SETTINGS})
        if s.get("voice_profile"):
            self.turns.model = SpeakerModel(s["voice_profile"])
        if s.get("speaker_mode") in ("auto", "manual"):
            self.turns.mode = s["speaker_mode"]

    async def on_frame(self, jpeg: bytes):
        if self._face_busy or self.faces is None:
            return                      # drop frames while busy: always work on the newest
        self._face_busy = True
        try:
            frame = decode_jpeg(jpeg)
            if frame is None:
                return
            found = await asyncio.get_running_loop().run_in_executor(FACE_POOL, self.faces.analyze, frame)
            st = self.identity.update(found)
            h, w = frame.shape[:2]
            if st.face_visible and st.person_id is not None:
                self.conv.person_frames[st.person_id] += 1
            await self.send({"type": "face", "visible": st.face_visible,
                             "bbox": _norm_bbox(st.bbox, w, h), "person_id": st.person_id if st.face_visible else None})
            if st.face_visible and st.person_id is not None and st.person_id != self._announced:
                self._announced = st.person_id
                await self._announce(st.person_id, st.just_registered, opener=True)
        finally:
            self._face_busy = False

    async def on_audio(self, pcm: bytes):
        for ev in self.turns.feed(pcm):
            t = ev["type"]
            if t == "speech_start":
                self.last_speech_at = time.time()
                # A cough used to cancel the suggestion being written and clear the dots.
                # Wait until the words arrive. A real line still ends it; a blip does not.
                await self.send({"type": "turn", "state": "speaking" if ev["speaker"] == "me" else "listening"})
            elif t == "false_start":
                # A short "uh" never becomes a transcript on its own. Read it when
                # it might be the wearer, so a few umms can open a suggestion.
                # The pause clock is unchanged either way.
                if self._worth_hesitation_probe(ev):
                    self._hearing += 1
                    await self._stt_q.put({
                        "type": "utterance", "audio": ev["audio"], "speaker": ev.get("speaker") or "me",
                        "source": "auto", "db": ev.get("db"), "f0": ev.get("f0"),
                        "hesitation_probe": True,
                    })
                elif self._pause_ready():
                    await self._flush_pending()
            elif t == "noise":
                # Fan, traffic, hiss, faint chatter: never sent to speech-to-text.
                log.info("ignored background sound (%.1f s, %s dB over room, pitch %s)",
                         ev.get("duration") or 0, ev.get("snr"), ev.get("pitched"))
                if self._pause_ready():
                    await self._flush_pending()
            elif t == "utterance":
                self.last_speech_at = time.time()
                self._hearing += 1
                await self._stt_q.put(ev)
            else:
                await self._turn_event(ev)
        self._check_stall()

    def _worth_hesitation_probe(self, ev: dict) -> bool:
        """A short blip that might be the wearer saying umm or ah."""
        audio = ev.get("audio")
        if audio is None or len(audio) == 0:
            return False
        if ev.get("speaker") == "me":
            return True
        me = (self.settings.get("voice_profile") or {}).get("me") or {}
        f0, mine = ev.get("f0"), me.get("f0")
        return bool(f0 and mine and semitones(float(f0), float(mine)) <= 4)

    async def _turn_event(self, ev: dict):
        if ev["type"] == "your_turn":
            if self.settings["turn_hints"]:
                await self.send({"type": "turn", "state": "your_turn", "text": "Your turn to speak"})
            await self._flush_pending()
        elif ev["type"] == "overlap" and self.settings["turn_hints"]:
            await self.send({"type": "turn", "state": "wait", "text": "Pause – let them finish"})

    async def control(self, msg: dict):
        t = msg.get("type")
        if t == "settings":
            self.update_settings(msg.get("settings", {}))
        elif t == "calibrate_start":
            self.turns.start_calibration()
        elif t == "calibrate_stop":
            prof = self.turns.finish_calibration()
            if prof is not None:
                self.settings["voice_profile"] = prof
            await self.send({"type": "calibrated", "profile": prof})
        elif t == "ptt":
            for ev in self.turns.press(bool(msg.get("down"))):
                await self._turn_event(ev)
        elif t == "fix_speaker":
            await self._fix_speaker(int(msg.get("index", -1)))
        elif t == "end_conversation":
            await self.end_conversation()
        elif t == "rename" and msg.get("person_id"):
            self.store.rename(int(msg["person_id"]), str(msg["name"]))
            await self._announce(int(msg["person_id"]), False)

    # ------------------------------------------------------------ pipeline
    async def _announce(self, pid: int, new: bool, *, opener: bool = False):
        if opener:
            # A different face. Drop a starter that belonged to the previous person.
            self._invalidate_opener()
            self._showing_cues = False
            self._shown_replies = []
            if self._placeholder:
                self._placeholder = False
                await self.send({"type": "cues", "cleared": True})
        p = self.store.get(pid)
        if not p:
            return
        self._announced = pid
        prof = p.profile or {}
        await self.send({
            "type": "person", "id": p.id, "name": p.name, "named": p.named,
            "new": new,
            "relationship": str(prof.get("relationship") or "").strip(),
            "designation": str(prof.get("designation") or "").strip(),
            "conversations": p.conversation_count,
        })
        if opener:
            self._start_opener(pid)

    def _invalidate_opener(self):
        self._opener_gen += 1
        task = self._opener_task
        self._opener_task = None
        if task is not None and not task.done():
            task.cancel()

    def _start_opener(self, pid: int):
        person = self.store.get(pid)
        if not llm.worth_opener(person) or self.conv.turns or self.turns.in_speech:
            return
        self._opener_gen += 1
        gen = self._opener_gen
        self._opener_task = asyncio.create_task(self._make_opener(pid, gen))

    def _real_speech_started(self) -> bool:
        """A cough, a backchannel, or a stall is not the conversation starting."""
        return any(not is_ignorable(t.get("text") or "") for t in self.conv.turns)

    async def _opener_ready(self, pid: int, gen: int) -> bool:
        """True once a cough has passed and nobody has actually spoken yet."""
        while True:
            if gen != self._opener_gen or self._announced != pid or self._real_speech_started():
                return False
            if not self.turns.in_speech and self._hearing == 0:
                return True
            await asyncio.sleep(0.05)

    async def _make_opener(self, pid: int, gen: int):
        try:
            await asyncio.sleep(0)                 # so a cancel that arrived early lands before we show dots
            if not await self._opener_ready(pid, gen):
                return
            self._placeholder = True
            await self.send({"type": "cues", "pending": True})
            if not await self._opener_ready(pid, gen):
                await self._drop_if_ours(gen)
                return
            person = self.store.get(pid)
            replies = await llm.opening_line(person, self.settings)
            if not await self._opener_ready(pid, gen):
                await self._drop_if_ours(gen)
                return
            self._placeholder = False
            self._shown_replies = list(replies)
            self._showing_cues = bool(replies)
            await self.send({"type": "cues", "replies": replies, "social": None, "emotion": None})
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("opener failed: %s", e)
            if gen == self._opener_gen:
                await self._drop_placeholder()
                await self.send({"type": "error", "where": "llm", "message": str(e)[:200]})

    async def _drop_if_ours(self, gen: int):
        """Clear the dots only if a newer starter, or a reply, has not taken the lens."""
        cue_running = self._cue_task is not None and not self._cue_task.done()
        if gen == self._opener_gen and not cue_running:
            await self._drop_placeholder()

    def _is_wearer_hesitation(self, ev: dict, text: str) -> bool:
        """A stall from the wearer. A quiet "um" is often labelled as the other person."""
        if not wearer_hesitating(text):
            return False
        if ev.get("speaker") == "me" or ev.get("source") in ("button", "calibration"):
            return True
        me = (self.settings.get("voice_profile") or {}).get("me") or {}
        f0, mine = ev.get("f0"), me.get("f0")
        if f0 and mine and semitones(float(f0), float(mine)) <= 4:
            return True
        return False

    async def _read_hesitation(self, ev: dict, loop) -> None:
        """Transcribe a short wearer blip. Only a real stall becomes a turn."""
        text, conf = await loop.run_in_executor(
            STT_POOL, self.stt.transcribe, ev["audio"], "um, uh, ah, hmm")
        if not text or not self._is_wearer_hesitation(ev, text):
            log.info("short sound ignored: %r", (text or "")[:40])
            return
        await self._accept_turn(ev, text, conf, loop_hesitation=True, stall_kept=True)

    async def _accept_turn(self, ev: dict, text: str, conf: float, *, loop_hesitation: bool = False,
                           stall_kept: bool = False):
        turn = {"speaker": "me" if loop_hesitation else ev["speaker"],
                "text": text, "conf": round(conf, 2), "t": time.time(),
                "db": ev.get("db"), "f0": ev.get("f0"), "source": ev.get("source", "auto"),
                "voiced_ms": ev.get("voiced_ms")}
        self.conv.turns.append(turn)
        spoken = {"type": "transcript", "index": len(self.conv.turns) - 1,
                  **{k: turn[k] for k in ("speaker", "text", "conf", "source")}}
        if wearer_hesitating(text):
            spoken["hesitation"] = True
        await self.send(spoken)
        if self.turns.model.calibrated:
            await self.send({"type": "voice_profile", "profile": self.turns.model.to_dict()})
        elif self.settings["speaker_mode"] == "auto" and len(self.conv.turns) == 1:
            await self.send({"type": "turn", "state": "info",
                             "text": "Calibrate your voice, or hold the talk button while you speak"})
        hesitating = self._is_wearer_hesitation(
            {**ev, "speaker": turn["speaker"]}, text)
        # Silence keeps running through an umm or a cough. A real line starts it over.
        if hesitating or is_ignorable(text):
            if not stall_kept:
                self.turns.add_stall_ms(int(round(float(ev.get("duration") or 0) * 1000)))
            if hesitating and self._stall_on:
                self._hesitation_hits += hesitation_hits(text)
        else:
            self._begin_stall()
        # Lines leave when the wearer answers (the offered line, or their own
        # words), or the subject changes. A cough, a stall, or more from the
        # other person on the same subject leaves them up.
        live = bool(self._shown_replies) or (
            self._cue_task is not None and not self._cue_task.done())
        ended = live and (not hesitating) and suggestion_expired(
            turn, self._shown_replies, anchor=self._anchor_text(),
            wearer=turn["speaker"] == "me")
        if ended:
            self._stale_after = len(self.conv.turns) - 2
            if self._pending and self._pending[0] <= self._stale_after:
                self._pending = None
            await self._dismiss_shown()
            self._hesitated = False
        if hesitating:
            self._hesitated = True
        elif turn["speaker"] == "them" and not loop_hesitation:
            self._hesitated = False
            self._consider_cues()
        elif ended:
            self._cancel_cues(clear_placeholder=True)
        self._check_stall()

    async def _stt_worker(self):
        loop = asyncio.get_running_loop()
        while True:
            ev = await self._stt_q.get()
            try:
                if ev.get("hesitation_probe"):
                    await self._read_hesitation(ev, loop)
                    continue
                # The last lines as a Whisper prompt is what got "hello" (me)
                # written back out as "hello" (them). Off unless asked for.
                context = (" ".join(t["text"] for t in self.conv.turns[-4:])
                           if config.STT_CONTEXT_PROMPT else "")
                text, conf = await loop.run_in_executor(STT_POOL, self.stt.transcribe, ev["audio"], context)
                if not text:
                    continue
                # A cough or some room noise is often transcribed as the line
                # just said, and then labelled as the other person.
                if replayed_as_other_speaker(text, ev.get("speaker") or "", self.conv.turns):
                    log.info("dropped replayed transcript (%s): %s", ev.get("speaker"), text[:80])
                    continue
                # Short lines too ("hello"), when the clip was faint or unsure.
                if self._weak_clip(ev, conf) and weak_echo(text, self.conv.turns, time.time()):
                    log.info("dropped echo from a weak clip (%s, %s dB, conf %.2f): %s",
                             ev.get("speaker"), ev.get("snr"), conf, text[:80])
                    continue
                # Generous: the whole span of the clip, not only its loudest frames.
                speech_ms = max(ev.get("voiced_ms") or 0,
                                (ev.get("duration") or 0) * 1000 - config.END_OF_TURN_MS)
                if too_many_words(text, speech_ms):
                    log.info("dropped invented text (%d ms of voice): %s", speech_ms, text[:80])
                    continue
                await self._accept_turn(ev, text, conf)
            except Exception:
                log.exception("STT failed")
            finally:
                self._hearing = max(0, self._hearing - 1)
                await self._release_pending()
                self._check_stall()

    @staticmethod
    def _weak_clip(ev: dict, conf: float) -> bool:
        """Barely above the room, or Whisper itself was unsure."""
        snr = ev.get("snr")
        return ((snr is not None and snr < config.STT_ECHO_WEAK_SNR_DB)
                or conf < config.STT_ECHO_WEAK_CONF)

    async def _release_pending(self):
        """Ask for a line that was held, once the extra audio has been heard.

        A cough or a stall from the wearer closes the floor (the last speaker
        is no longer the partner), so the usual pause never arrives. The line
        they already finished is still worth answering.
        """
        if not (self._hearing == 0 and self._stt_q.empty() and self._pending):
            return
        if self.turns.in_speech:
            return
        if not self._pending_still_wanted():
            self._pending = None
            return
        if self._pause_ready() or self.turns.last_speaker == "me":
            await self._flush_pending()

    def _pending_still_wanted(self) -> bool:
        pending = self._pending
        if not pending:
            return False
        idx, kind = pending
        if self._stale_after is not None and idx <= self._stale_after:
            return False
        if kind == "unclear":
            return True
        return self._later_turns_keep_the_moment(idx)

    async def _fix_speaker(self, i: int):
        """Wearer tapped a transcript line: flip me/them and teach the voice model."""
        if not 0 <= i < len(self.conv.turns):
            return
        t = self.conv.turns[i]
        t["speaker"] = "them" if t["speaker"] == "me" else "me"
        t["source"] = "corrected"
        self.turns.correct(t["db"], t["f0"], t["speaker"])
        await self.send({"type": "transcript_fixed", "index": i, "speaker": t["speaker"]})
        await self.send({"type": "voice_profile", "profile": self.turns.model.to_dict()})
        # Wearer corrected the latest line to Them and nobody is talking: that is a real moment.
        if t["speaker"] == "them" and i == len(self.conv.turns) - 1:
            self._consider_cues(force=not self.turns.in_speech)

    def _pause_ready(self) -> bool:
        """The silence required before a suggestion. The long-pause toggle waits longer."""
        if self.turns.in_speech:
            return False
        if self.settings.get("long_pause"):
            return self.turns.long_pause_open
        return self.turns.floor_open

    def _anchor_text(self) -> str:
        """The line the suggestion on screen (or on the way) is answering."""
        idx = self._cued_for
        if idx is not None and 0 <= idx < len(self.conv.turns):
            return self.conv.turns[idx].get("text") or ""
        if self._shown_replies:
            return " ".join(self._shown_replies)
        return ""

    def _suggestion_locked(self, turn: dict) -> bool:
        """A line already showing, or still being written, stays until the wearer answers or the topic changes."""
        inflight = (self._cue_task is not None and not self._cue_task.done()
                    and self._cued_for is not None)
        if not self._shown_replies and not inflight:
            return False
        return not suggestion_expired(turn, self._shown_replies, anchor=self._anchor_text())

    def _begin_stall(self):
        """A real line landed. The quiet clock starts over from the silence after it."""
        self.turns.reset_stall()
        self._stall_on = True
        self._stall_gen += 1
        self._reply_sent = False
        self._ice_sent = False
        self._stall_reply = None
        self._stall_ice = None
        self._stall_emotion = None
        self._hesitation_hits = 0

    def _unanswered_prompt(self) -> tuple[int, str] | None:
        """The last real thing they said that the wearer has not answered yet."""
        for i in range(len(self.conv.turns) - 1, -1, -1):
            if self._stale_after is not None and i <= self._stale_after:
                return None
            turn = self.conv.turns[i]
            text = turn.get("text") or ""
            if is_ignorable(text):
                continue
            if turn.get("speaker") != "them":
                return None
            moment = cue_decision(turn)[0]
            return (i, moment) if moment else None

    def _check_stall(self):
        """Reply after 5s of quiet or a few umms. Ice breaker if it is still stuck at 8s."""
        if not self._stall_on or self.turns.in_speech or self._hearing > 0 or not self._stt_q.empty():
            return
        if int(self.settings.get("num_replies") or 0) <= 0:
            return
        elapsed = self.turns.stall_ms
        ice_at = int(round(config.CUE_ICEBREAKER_AFTER_S * 1000))
        reply_at = int(round(config.CUE_REPLY_AFTER_S * 1000))
        if not self._ice_sent and elapsed >= ice_at:
            self._ice_sent = True
            self._reply_sent = True
            log.info("ice breaker after %.1fs quiet", elapsed / 1000)
            self._schedule_icebreaker()
            return
        if self._reply_sent or (elapsed < reply_at and self._hesitation_hits < config.CUE_HESITATION_HITS):
            return
        found = self._unanswered_prompt()
        if not found:
            return
        idx, moment = found
        if self._suggestion_locked(self.conv.turns[idx]):
            self._reply_sent = True
            return
        self._reply_sent = True
        log.info("reply after a stall (%.1fs, %d hesitations)", elapsed / 1000, self._hesitation_hits)
        self._schedule_cues(idx, moment, force=True, single=True)

    def _consider_cues(self, force: bool = False):
        """A direct reply waits on the stall clock. Speaker corrections skip the wait."""
        if not self.conv.turns:
            return
        idx = len(self.conv.turns) - 1
        turn = self.conv.turns[idx]
        if turn["speaker"] != "them":
            return
        if self._suggestion_locked(turn) and not force:
            log.info("suggestion stays (topic unchanged): %s", turn["text"][:80])
            return
        if not force and (not self._stt_q.empty() or self.turns.in_speech):
            # The line is in the transcript. The stall clock asks once the extra audio is read.
            return
        moment, why = cue_decision(turn)
        if moment is None:
            text = turn.get("text") or ""
            # A cough or a fragment must not replace the lines with "didn't catch that".
            if is_ignorable(text):
                log.info("cue skipped (%s): %s", why, text[:80])
                return
            if needs_repeat(turn) and (force or self.turns.floor_open):
                self._send_unclear(idx)
            elif needs_repeat(turn) and not self.turns.in_speech:
                self._pending = (idx, "unclear")
            else:
                log.info("cue skipped (%s): %s", why, turn["text"][:80])
            return
        if force:
            # They corrected who spoke. Show a reply now, and start the 8s clock from here.
            self.turns.restart_stall_now()
            self._stall_on = True
            self._stall_gen += 1
            self._ice_sent = False
            self._hesitation_hits = 0
            self._reply_sent = True
            self._schedule_cues(idx, moment, True)
            return
        log.info("cue waiting out the stall (%s): %s", why, turn["text"][:80])

    def _later_turns_keep_the_moment(self, idx: int) -> bool:
        """A cough, a backchannel, or the same subject does not end the line we are answering."""
        if not 0 <= idx < len(self.conv.turns):
            return False
        anchor = self.conv.turns[idx].get("text") or ""
        for t in self.conv.turns[idx + 1:]:
            text = t.get("text") or ""
            if t.get("speaker") == "me" and wearer_hesitating(text):
                continue
            if suggestion_expired(
                    t, self._shown_replies, anchor=anchor, wearer=t.get("speaker") == "me"):
                return False
        return True

    def _cue_still_valid(self, idx: int) -> bool:
        if self._cued_for != idx or not (0 <= idx < len(self.conv.turns)):
            return False
        if not self._later_turns_keep_the_moment(idx):
            return False
        if self._stale_after is not None and idx <= self._stale_after:
            return False
        return cue_decision(self.conv.turns[idx])[0] is not None

    async def _await_worth(self, idx: int, force: bool) -> bool:
        """Wait out a cough or a line still being transcribed. False if the moment ended."""
        while True:
            if not self._cue_still_valid(idx):
                return False
            if self.turns.in_speech or self._hearing > 0:
                await asyncio.sleep(0.05)
                continue
            if force or self._pause_ready():
                return True
            # A later cough, stall, or same-topic line did not end the moment.
            if self.conv.turns[idx + 1:] and self._later_turns_keep_the_moment(idx):
                return True
            # The wearer rustled or stalled, so the floor clock did not reopen.
            # That sound is not a reason to drop the line.
            if self.turns.last_speaker == "me" and self._later_turns_keep_the_moment(idx):
                return True
            await asyncio.sleep(0.05)

    async def _flush_pending(self):
        """'Didn't catch that' still follows the short pause. Replies wait on the stall clock."""
        pending = self._pending
        if not pending or self.turns.in_speech or pending[1] != "unclear":
            return
        self._pending = None
        idx = pending[0]
        if not self._later_turns_keep_the_moment(idx):
            return
        self._send_unclear(idx)

    def _send_unclear(self, idx: int):
        if self._unclear_for == idx:
            return
        if self._shown_replies:
            return
        self._unclear_for = idx
        self._pending = None
        self._shown_replies = []
        self._showing_cues = True
        log.info("cue skipped (couldn't hear a real line)")
        asyncio.create_task(self.send({
            "type": "cues", "replies": [], "emotion": None,
            "social": "Didn't catch that – ask them to repeat"}))

    def _schedule_cues(self, idx: int, moment: str, force: bool, single: bool = False):
        if self._cued_for == idx and self._cue_task and not self._cue_task.done():
            return
        # A reply owns the dots from here. A starter still in flight must not clear them.
        self._invalidate_opener()
        self._pending = None
        if self._cue_task and not self._cue_task.done():
            self._cue_task.cancel()
        self._cue_gen += 1
        gen = self._cue_gen
        self._cued_for = idx
        self._cue_task = asyncio.create_task(self._make_cues(idx, moment, force, gen, single))

    def _schedule_icebreaker(self):
        """Add one new-topic line under the reply already on screen."""
        self._invalidate_opener()
        task = self._ice_task
        if task is not None and not task.done():
            task.cancel()
        self._ice_task = asyncio.create_task(self._make_icebreaker(self._stall_gen))

    def _cancel_cues(self, clear_placeholder: bool):
        """Stop an in-flight suggestion. A suggestion already on screen stays there."""
        self._pending = None
        task = self._cue_task
        running = task is not None and not task.done()
        if running:
            task.cancel()
            self._cued_for = None
        self._cue_task = None
        ice = self._ice_task
        if ice is not None and not ice.done():
            ice.cancel()
        self._ice_task = None
        self._invalidate_opener()
        if clear_placeholder and self._placeholder:
            self._placeholder = False
            asyncio.create_task(self.send({"type": "cues", "cleared": True}))

    async def _dismiss_shown(self):
        """Take the reply lines off the lens. The person's memory card stays."""
        if not self._showing_cues:
            return
        self._showing_cues = False
        self._shown_replies = []
        await self.send({"type": "cues", "dismissed": True})

    async def _drop_placeholder(self, gen: int | None = None):
        """Clear the dots unless a newer suggestion has already taken them."""
        if gen is not None and gen != self._cue_gen:
            return
        if self._placeholder:
            self._placeholder = False
            await self.send({"type": "cues", "cleared": True})

    async def _stall_quiet(self, stall_gen: int) -> bool:
        """True once nobody is mid-word and this stall is still the current one."""
        while self.turns.in_speech or self._hearing > 0:
            if stall_gen != self._stall_gen:
                return False
            await asyncio.sleep(0.05)
        return stall_gen == self._stall_gen

    def _stall_lines(self) -> list[str]:
        """The 5s reply, then the 8s ice breaker if it has arrived."""
        lines = []
        for line in (self._stall_reply, self._stall_ice):
            if line and line not in lines:
                lines.append(line)
        return lines

    async def _make_icebreaker(self, stall_gen: int):
        try:
            if not await self._stall_quiet(stall_gen):
                return
            # Leave the 5s line up. Dots would hide it while this one is written.
            pid = self.conv.person_id or self.identity.current
            person = self.store.get(pid) if pid else None
            self._last_cue_at = time.time()
            cues = await llm.icebreaker(
                person, self.conv.turns, {**self.settings, "num_replies": 1})
            if not await self._stall_quiet(stall_gen):
                return
            fresh = [r for r in (cues.get("replies") or []) if r]
            self._stall_ice = fresh[0] if fresh else None
            lines = self._stall_lines()
            self._shown_replies = lines
            self._showing_cues = bool(lines or cues.get("social"))
            await self.send({
                "type": "cues",
                "replies": lines,
                "social": cues.get("social"),
                "emotion": self._stall_emotion,
                "kind": "icebreaker",
            })
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("ice breaker failed: %s", e)
            await self.send({"type": "error", "where": "llm", "message": str(e)[:200]})

    async def _make_cues(self, idx: int, moment: str, force: bool, gen: int, single: bool = False):
        sent = False
        try:
            wait = config.CUE_MIN_INTERVAL_S - (time.time() - self._last_cue_at)
            if wait > 0:
                await asyncio.sleep(wait)
            if gen != self._cue_gen or not await self._await_worth(idx, force):
                await self._drop_placeholder(gen)
                return
            self._placeholder = True
            await self.send({"type": "cues", "pending": True})
            if gen != self._cue_gen or not await self._await_worth(idx, force):
                await self._drop_placeholder(gen)
                return
            pid = self.conv.person_id or self.identity.current
            person = self.store.get(pid) if pid else None
            self._last_cue_at = time.time()
            text = self.conv.turns[idx]["text"]
            log.info("cue on %s: %s", moment, text[:80])
            settings = {**self.settings, "num_replies": 1} if single else self.settings
            cues = await llm.live_cues(person, self.conv.turns, settings,
                                       moment=MOMENT_TEXT.get(moment, moment))
            if gen != self._cue_gen or not await self._await_worth(idx, force):
                await self._drop_placeholder(gen)
                return
            sent = True
            self._placeholder = False
            replies = [r for r in (cues.get("replies") or []) if r]
            if single:
                self._stall_reply = replies[0] if replies else None
                self._stall_emotion = cues.get("emotion")
                replies = self._stall_lines()
            else:
                self._stall_reply = None
                self._stall_ice = None
                self._stall_emotion = cues.get("emotion")
            cues = {**cues, "replies": replies}
            self._shown_replies = replies
            self._showing_cues = bool(replies or cues.get("social"))
            await self.send({"type": "cues", **cues})
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("cue generation failed: %s", e)
            await self._drop_placeholder(gen)
            await self.send({"type": "error", "where": "llm", "message": str(e)[:200]})
        finally:
            if not sent and self._cue_gen == gen and self._cued_for == idx:
                self._cued_for = None

    async def end_conversation(self):
        self._cancel_cues(clear_placeholder=True)
        self._showing_cues = False
        self._shown_replies = []
        self._hesitated = False
        self._stale_after = None
        self._stall_on = False
        self._stall_gen += 1
        self._reply_sent = False
        self._ice_sent = False
        self._stall_reply = None
        self._stall_ice = None
        self._stall_emotion = None
        self._hesitation_hits = 0
        conv, self.conv = self.conv, Conversation()
        self._announced = None
        if len(conv.turns) < config.MIN_TURNS_TO_SUMMARIZE:
            return
        pid = conv.person_id
        person = self.store.get(pid) if pid else None
        try:
            res = await llm.summarize(person, conv.turns)
        except Exception as e:
            log.warning("summary failed, saving transcript only: %s", e)
            res = {"record": {"summary": "", "title": "Conversation (summary unavailable)"},
                   "name": None, "profile": person.profile if person else {}}
        cid = self.store.save_conversation(pid, conv.started, time.time(), conv.turns,
                                           res["record"], res["profile"])
        if person and res["name"] and not person.named:
            self.store.rename(pid, res["name"])
        await self.send({"type": "conversation_saved", "person_id": pid,
                         "name": res["name"] or (person.name if person else None),
                         "conversation_id": cid, "title": res["record"].get("title", ""),
                         "summary": res["record"].get("summary", "")})

    async def _watchdog(self):
        while True:
            await asyncio.sleep(2)
            now = time.time()
            if (self.conv.turns
                    and now - self.identity.last_face_at > config.SESSION_END_NO_FACE_S
                    and now - self.last_speech_at > config.SESSION_END_SILENCE_S):
                await self.end_conversation()

    async def close(self):
        for t in self._tasks:
            t.cancel()
        await self.end_conversation()


def _norm_bbox(b, w, h):
    if not b:
        return None
    x, y, bw, bh = b
    return [round(x / w, 4), round(y / h, 4), round(bw / w, 4), round(bh / h, 4)]
