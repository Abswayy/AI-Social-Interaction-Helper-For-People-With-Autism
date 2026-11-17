"""Speech-to-text.

Default: hosted Whisper over an OpenAI-compatible API (Groq by default:
free tier, nothing to download). Set STT_PROVIDER=local to use faster-whisper
on the machine instead (needs `pip install faster-whisper`, ~0.5 GB model).

Accuracy guards, because a wrong transcript produces a wrong AI cue:
  * only speech that clears the noise gate in audio.py is sent (no
    transcribing fan noise or faint chatter, fewer API calls)
  * the conversation so far is NOT sent as a Whisper prompt by default:
    on a quiet or noisy clip Whisper writes the prompt back out, which is
    how "hello" from the wearer came back as "hello" from the other person
    (STT_CONTEXT_PROMPT=1 turns it back on)
  * faint audio is not blown up to full volume before Whisper hears it
  * low-confidence segments and Whisper's known hallucinations are dropped,
    and every result carries a confidence so cues can skip shaky text
  * a weak clip that repeats a recent line, or far more words than the
    clip had voice for, is treated as invented and dropped
"""
from __future__ import annotations

import difflib
import io
import logging
import math
import re
import threading
import wave

import httpx
import numpy as np

from . import config

log = logging.getLogger("stt")

HALLUCINATIONS = re.compile(
    r"^(thanks? (you )?for watching.*|subscribe|please subscribe|you|bye\.?|"
    r"\[.*\]|\(.*\)|♪.*|thank you\.?|\.+|okay\.?|"
    # what Whisper writes on silence, hiss and room noise
    r"thank you (so|very) much( for watching)?[.!]?|"
    r"(please )?(like and )?subscribe.*|(subtitles?|captions?|transcri(bed|ption)) by.*|"
    r".*amara\.org.*|you you( you)*\.?)$", re.I)


def _hesitation_segment(text: str) -> bool:
    """A stall like "uh" is often scored as non-speech. Keep it anyway."""
    from .cues import wearer_hesitating
    return wearer_hesitating(text)


def filter_segments(segments: list[dict]) -> tuple[str, float]:
    """segments: dicts with text, avg_logprob, no_speech_prob, compression_ratio."""
    kept, probs = [], []
    for s in segments:
        t = (s.get("text") or "").strip()
        # "umm" / "uh" / "ah" fail the speech score often enough that the
        # hesitation feature looked random. A stall is still a real token.
        if t and _hesitation_segment(t):
            kept.append(t)
            probs.append(max(0.55, math.exp(min(0.0, s.get("avg_logprob", -0.5)))))
            continue
        if s.get("no_speech_prob", 0) > config.STT_MAX_NO_SPEECH:
            continue
        if s.get("avg_logprob", 0) < config.STT_MIN_AVG_LOGPROB:
            continue
        if s.get("compression_ratio", 1) > 2.4:          # repetitive loop = hallucination
            continue
        if not t or HALLUCINATIONS.match(t):
            continue
        kept.append(t)
        probs.append(math.exp(s.get("avg_logprob", -0.3)))
    return " ".join(kept).strip(), (float(np.mean(probs)) if probs else 0.0)


def _norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _is_repeated_copy(words: list[str], original: list[str]) -> bool:
    """True when `words` is `original`, or that line pasted back to back."""
    n = len(original)
    if n < 3 or len(words) < n or len(words) % n != 0:
        return False
    return all(words[i:i + n] == original for i in range(0, len(words), n))


def echoes_recent_line(text: str, recent: list[str]) -> bool:
    """True when `text` is a recent line copied back, not something new.

    The speech model is given the last lines as a prompt. On a cough, reverb,
    or a quiet noise it sometimes writes that prompt out as the transcript.
    """
    words = _norm_words(text)
    if len(words) < 3:
        return False
    pieces = [p for p in (_norm_words(r) for r in recent) if p]
    for prev in pieces:
        if _is_repeated_copy(words, prev):
            return True
    if len(pieces) >= 2:
        joined: list[str] = []
        for piece in pieces:
            joined.extend(piece)
        if _is_repeated_copy(words, joined):
            return True
    return False


def replayed_as_other_speaker(text: str, speaker: str, turns: list[dict]) -> bool:
    """The clip was transcribed as a line the other person already said.

    The same person repeating themselves is kept. A copy credited to the
    other speaker is the prompt being played back, not a second utterance.
    """
    recent = [t for t in turns[-6:] if (t.get("text") or "").strip()]
    if not recent or not echoes_recent_line(text, [t.get("text") or "" for t in recent]):
        return False
    words = _norm_words(text)
    for turn in recent:
        if turn.get("speaker") == speaker:
            continue
        if _is_repeated_copy(words, _norm_words(turn.get("text") or "")):
            return True
    if len(recent) >= 2:
        joined: list[str] = []
        for turn in recent:
            joined.extend(_norm_words(turn.get("text") or ""))
        if _is_repeated_copy(words, joined):
            return True
    return False


def _same_line(new: list[str], old: list[str]) -> bool:
    """`new` is `old`, nearly `old`, or a 3+ word piece of it."""
    if not new or not old:
        return False
    if new == old:
        return True
    if len(new) < 3 or len(old) < 3:
        return False
    if difflib.SequenceMatcher(None, new, old, autojunk=False).ratio() >= 0.8:
        return True
    n = len(new)
    return any(old[i:i + n] == new for i in range(len(old) - n + 1))


def weak_echo(text: str, turns: list[dict], now: float, window_s: float | None = None) -> bool:
    """A faint or uncertain clip came back as a line that was just said.

    Catches what replayed_as_other_speaker cannot: one- and two-word lines
    ("hello"), near copies, a piece of the line, and a copy credited to the
    same speaker. Call it only for weak clips, so a clear "hello" back from
    the other person is still kept.
    """
    words = _norm_words(text)
    if not words:
        return False
    window = config.STT_ECHO_WINDOW_S if window_s is None else window_s
    for turn in reversed(turns[-4:]):
        if now - float(turn.get("t") or 0) > window:
            break
        if _same_line(words, _norm_words(turn.get("text") or "")):
            return True
    return False


def too_many_words(text: str, voiced_ms: float | None) -> bool:
    """Whisper wrote a sentence for a clip that only had a moment of voice in it."""
    if not voiced_ms:
        return False
    limit = 3 + config.STT_MAX_WORDS_PER_S * float(voiced_ms) / 1000
    return len(_norm_words(text)) > limit


def drop_prompt_echo(text: str, context: str) -> str:
    """Remove a transcript that is only the conditioning prompt."""
    cleaned = (text or "").strip()
    if context and echoes_recent_line(cleaned, [context]):
        log.info("dropped prompt echo: %s", cleaned[:80])
        return ""
    return cleaned


def _preprocess(audio: np.ndarray) -> np.ndarray:
    """Remove DC offset and bring a quiet voice up, by at most STT_MAX_GAIN_DB.

    Scaling every clip to full volume made faint room noise sound like a loud
    voice, and Whisper then "heard" words in it.
    """
    if audio.size == 0:
        return audio
    out = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(out))) or 1.0
    gain = min(0.9 / peak, 10 ** (config.STT_MAX_GAIN_DB / 20))
    return (out * gain).astype(np.float32)


def to_wav(audio: np.ndarray) -> bytes:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(config.SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class ApiTranscriber:
    """Whisper via /audio/transcriptions (Groq, OpenAI, or any compatible server)."""

    def __init__(self):
        if not config.STT_API_KEY:
            raise RuntimeError("Set GROQ_API_KEY (or STT_API_KEY) in .env – see .env.example")
        self.http = httpx.Client(base_url=config.STT_API_BASE_URL.rstrip("/"), timeout=15,
                                 headers={"Authorization": f"Bearer {config.STT_API_KEY}"})
        log.info("Speech-to-text: %s @ %s", config.STT_API_MODEL, config.STT_API_BASE_URL)

    def transcribe(self, audio: np.ndarray, context: str = "") -> tuple[str, float]:
        data = {"model": config.STT_API_MODEL, "response_format": "verbose_json", "temperature": "0"}
        if config.STT_LANGUAGE:
            data["language"] = config.STT_LANGUAGE
        if context:
            data["prompt"] = context[-400:]
        files = {"file": ("speech.wav", to_wav(_preprocess(audio)), "audio/wav")}
        for attempt in range(2):
            r = self.http.post("/audio/transcriptions", data=data, files=files)
            if r.status_code == 429 and attempt == 0:      # rate limited: brief back-off, once
                import time; time.sleep(1.5)
                continue
            r.raise_for_status()
            break
        body = r.json()
        segments = body.get("segments")
        if segments is None:          # some providers return text only
            text = (body.get("text") or "").strip()
            if HALLUCINATIONS.match(text or "."):
                return "", 0.0
            return drop_prompt_echo(text, context), 0.8
        text, conf = filter_segments(segments)
        return drop_prompt_echo(text, context), conf


class LocalTranscriber:
    """faster-whisper on this machine (offline; downloads a model on first run)."""

    def __init__(self):
        from faster_whisper import WhisperModel
        device = config.STT_DEVICE
        if device == "auto":
            try:
                import ctranslate2
                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                device = "cpu"
        compute = "float16" if device == "cuda" and config.STT_COMPUTE == "int8" else config.STT_COMPUTE
        log.info("Loading local Whisper %s on %s (%s)", config.STT_MODEL, device, compute)
        self.model = WhisperModel(config.STT_MODEL, device=device, compute_type=compute,
                                  download_root=str(config.MODELS_DIR / "whisper"))
        self._lock = threading.Lock()

    def transcribe(self, audio: np.ndarray, context: str = "") -> tuple[str, float]:
        with self._lock:
            segments, _ = self.model.transcribe(
                _preprocess(audio), language=config.STT_LANGUAGE, beam_size=5,
                vad_filter=True, vad_parameters={"min_silence_duration_ms": 300},
                condition_on_previous_text=False, initial_prompt=context[-400:] or None,
                temperature=[0.0, 0.2, 0.4], no_speech_threshold=config.STT_MAX_NO_SPEECH)
            segs = [{"text": s.text, "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
                     "compression_ratio": s.compression_ratio} for s in segments]
        text, conf = filter_segments(segs)
        return drop_prompt_echo(text, context), conf


def make_transcriber():
    return LocalTranscriber() if config.STT_PROVIDER == "local" else ApiTranscriber()
