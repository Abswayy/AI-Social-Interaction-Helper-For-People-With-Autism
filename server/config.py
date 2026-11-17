"""All tunables in one place. Override any value with an environment variable
of the same name (e.g. LLM_MODEL=llama3.1:8b python -m server.main)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load KEY=value lines from .env (no extra dependency). Real env vars win.
_dotenv = ROOT / ".env"
if _dotenv.exists():
    for _line in _dotenv.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))


def _env(name, default, cast=str):
    v = os.getenv(name)
    if v is None:
        return default
    if cast is bool:
        return v.lower() in ("1", "true", "yes", "on")
    return cast(v)


DATA_DIR = Path(_env("DATA_DIR", str(ROOT / "data")))
MODELS_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "glasses.db"

# ---------- Database ----------
# Empty = local SQLite file (laptop demo). For glasses / cloud: a Postgres URL, e.g.
# postgresql://postgres:PASSWORD@db.xxxx.supabase.co:5432/postgres
DATABASE_URL = _env("DATABASE_URL", "")

# ---------- Server ----------
HOST = _env("HOST", "0.0.0.0")
PORT = _env("PORT", 8000, int)

# ---------- Face ----------
FACE_MIN_SIZE_PX = _env("FACE_MIN_SIZE_PX", 70, int)      # ignore tiny/far faces
FACE_DET_SCORE = _env("FACE_DET_SCORE", 0.85, float)
FACE_MATCH_THRESHOLD = _env("FACE_MATCH_THRESHOLD", 0.42, float)  # SFace cosine; opencv suggests 0.363
FACE_MAX_EMBEDDINGS = _env("FACE_MAX_EMBEDDINGS", 8, int)  # gallery per person
AUTO_REGISTER = _env("AUTO_REGISTER", True, bool)
AUTO_REGISTER_FRAMES = _env("AUTO_REGISTER_FRAMES", 5, int)  # consistent sightings needed
AUTO_REGISTER_MIN_SECONDS = _env("AUTO_REGISTER_MIN_SECONDS", 1.5, float)

# ---------- Audio / turn-taking ----------
SAMPLE_RATE = 16000
VAD_AGGRESSIVENESS = _env("VAD_AGGRESSIVENESS", 3, int)   # 0-3, 3 = most noise-robust
END_OF_TURN_MS = _env("END_OF_TURN_MS", 700, int)        # silence that closes an utterance
YOUR_TURN_MS = _env("YOUR_TURN_MS", 1200, int)           # other person's pause => "your turn"
LONG_PAUSE_MS = _env("LONG_PAUSE_MS", 3200, int)         # "only after a long pause" toggle
MIN_UTTERANCE_MS = _env("MIN_UTTERANCE_MS", 400, int)
MAX_UTTERANCE_S = _env("MAX_UTTERANCE_S", 20, int)
# Noise gate. The room's background level is tracked all the time (a fan or
# traffic becomes the new "quiet" within ~2.5 s). A sound only counts as speech
# when it is clearly above that level AND has a voice pitch in it.
#   Noise still getting through  -> raise SPEECH_MIN_SNR_DB (e.g. 11)
#   Quiet partner being missed   -> lower SPEECH_MIN_SNR_DB (e.g. 6)
NOISE_FLOOR_MIN_DB = _env("NOISE_FLOOR_MIN_DB", -70.0, float)    # dBFS; quieter rooms are treated as this
NOISE_FLOOR_MAX_DB = _env("NOISE_FLOOR_MAX_DB", -30.0, float)    # dBFS; cap so a loud voice never becomes "background"
SPEECH_FRAME_SNR_DB = _env("SPEECH_FRAME_SNR_DB", 6.0, float)    # a 20 ms frame must be this far above the room
SPEECH_MIN_SNR_DB = _env("SPEECH_MIN_SNR_DB", 8.0, float)        # a whole line must be this far above the room
SPEECH_MIN_PITCHED = _env("SPEECH_MIN_PITCHED", 0.2, float)      # share of the line that has a voice pitch (fans/hiss have none)
# Speaker attribution: the glasses mic is ~5cm from the wearer's mouth, so the
# wearer is much louder than the other person. Calibrated per user in the UI.
SELF_VOICE_MARGIN_DB = _env("SELF_VOICE_MARGIN_DB", 6.0, float)

# One key for both speech and AI when using Groq (free at console.groq.com).
GROQ_API_KEY = _env("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1"

# ---------- Speech-to-text ----------
STT_PROVIDER = _env("STT_PROVIDER", "api")      # api (nothing to download) | local (faster-whisper)
STT_API_BASE_URL = _env("STT_API_BASE_URL", GROQ_URL)          # or https://api.openai.com/v1
STT_API_KEY = _env("STT_API_KEY", GROQ_API_KEY)
STT_API_MODEL = _env("STT_API_MODEL", "whisper-large-v3")      # OpenAI: whisper-1
STT_LANGUAGE = _env("STT_LANGUAGE", "en") or None   # "" = auto-detect (e.g. Hindi)
STT_MAX_NO_SPEECH = _env("STT_MAX_NO_SPEECH", 0.6, float)
STT_MIN_AVG_LOGPROB = _env("STT_MIN_AVG_LOGPROB", -1.0, float)
STT_WORKERS = _env("STT_WORKERS", 3, int)       # parallel API calls across all users
# Sending the last lines to Whisper as a "prompt" helps spelling, but on a quiet
# or noisy clip Whisper writes the prompt back out ("hello" from you came back
# as "hello" from them). Off by default.
STT_CONTEXT_PROMPT = _env("STT_CONTEXT_PROMPT", False, bool)
STT_MAX_GAIN_DB = _env("STT_MAX_GAIN_DB", 30.0, float)          # don't blow faint noise up to full volume
# A clip this weak (dB above the room, or Whisper confidence) that repeats a
# recent line is treated as an echo, not a new line.
STT_ECHO_WEAK_SNR_DB = _env("STT_ECHO_WEAK_SNR_DB", 15.0, float)
STT_ECHO_WEAK_CONF = _env("STT_ECHO_WEAK_CONF", 0.5, float)
STT_ECHO_WINDOW_S = _env("STT_ECHO_WINDOW_S", 10.0, float)
STT_MAX_WORDS_PER_S = _env("STT_MAX_WORDS_PER_S", 7.0, float)   # more words than this per second of voice = invented
# only for STT_PROVIDER=local
STT_MODEL = _env("STT_MODEL", "small")
STT_DEVICE = _env("STT_DEVICE", "auto")
STT_COMPUTE = _env("STT_COMPUTE", "int8")

# ---------- LLM (any OpenAI-compatible API: Groq, OpenAI, Together, Ollama...) ----------
LLM_BASE_URL = _env("LLM_BASE_URL", GROQ_URL)
LLM_API_KEY = _env("LLM_API_KEY", GROQ_API_KEY)
LLM_MODEL = _env("LLM_MODEL", "openai/gpt-oss-120b")          # live cues (llama-3.3-70b was retired Aug 2026)
LLM_REASONING_EFFORT = _env("LLM_REASONING_EFFORT", "low")   # for reasoning models: low = fastest
LLM_SUMMARY_MODEL = _env("LLM_SUMMARY_MODEL", "")              # "" = same as LLM_MODEL
LLM_CHECK_MODEL = _env("LLM_CHECK_MODEL", "")                  # hallucination check; "" = same as LLM_MODEL
LLM_TIMEOUT_S = _env("LLM_TIMEOUT_S", 20, float)
LLM_MAX_CONCURRENT = _env("LLM_MAX_CONCURRENT", 4, int)        # global cap across users
CUE_CONTEXT_TURNS = _env("CUE_CONTEXT_TURNS", 12, int)         # rolling window sent for cues
CUE_MIN_INTERVAL_S = _env("CUE_MIN_INTERVAL_S", 2.0, float)
# A reply waits until the wearer has been stuck: this much quiet, or a few
# "umm" / "ah" sounds. If the talk still has not moved on, an ice breaker.
CUE_REPLY_AFTER_S = _env("CUE_REPLY_AFTER_S", 5.0, float)
CUE_ICEBREAKER_AFTER_S = _env("CUE_ICEBREAKER_AFTER_S", 8.0, float)
CUE_HESITATION_HITS = _env("CUE_HESITATION_HITS", 2, int)      # a couple of umms opens the reply early
# The line itself still has to be worth answering.
CUE_MIN_CONF = _env("CUE_MIN_CONF", 0.40, float)               # shakier than this: stay quiet
CUE_REPEAT_BELOW = _env("CUE_REPEAT_BELOW", 0.35, float)       # long line this shaky: ask them to repeat
CUE_MIN_WORDS = _env("CUE_MIN_WORDS", 3, int)                  # statements shorter than this are not answered
CUE_MIN_QUESTION_WORDS = _env("CUE_MIN_QUESTION_WORDS", 2, int)

# ---------- Conversation lifecycle ----------
SESSION_END_NO_FACE_S = _env("SESSION_END_NO_FACE_S", 25, float)
SESSION_END_SILENCE_S = _env("SESSION_END_SILENCE_S", 20, float)
MIN_TURNS_TO_SUMMARIZE = _env("MIN_TURNS_TO_SUMMARIZE", 2, int)

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
