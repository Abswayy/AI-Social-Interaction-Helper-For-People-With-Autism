"""LLM calls through any OpenAI-compatible endpoint (Groq by default).

Point LLM_BASE_URL at OpenAI, Together, Ollama, vLLM, etc. without code changes.
Providers retire model IDs regularly, so at startup we check the configured
model exists and fall back to a known-good one instead of failing with a 404.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re

import httpx

from . import config, prompts

log = logging.getLogger("llm")
_sem = asyncio.Semaphore(config.LLM_MAX_CONCURRENT)   # global cap = backpressure under load
_client: httpx.AsyncClient | None = None


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=config.LLM_BASE_URL.rstrip("/"), timeout=config.LLM_TIMEOUT_S,
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}"})
    return _client


class LLMError(RuntimeError):
    pass


# Tried in order if the configured model is missing from the provider's list.
FALLBACK_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b",
                   "gpt-4o-mini", "gpt-4.1-mini"]
FALLBACK_STT = ["whisper-large-v3-turbo", "whisper-large-v3", "whisper-1"]


def is_reasoning(model: str) -> bool:
    m = model.lower()
    return "gpt-oss" in m or m.startswith(("o1", "o3", "o4")) or "qwen3" in m


async def chat_json(system: str, user: str, model: str | None = None,
                    max_tokens: int = 400, temperature: float = 0.4) -> dict:
    model = model or config.LLM_MODEL
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature, "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if is_reasoning(model):
        # Reasoning tokens count against max_tokens: keep thinking short, leave room for the answer.
        body["reasoning_effort"] = getattr(config, "LLM_REASONING_EFFORT", "low")
        body["max_tokens"] = max_tokens + 600
    async with _sem:
        r = await _http().post("/chat/completions", json=body)
        if r.status_code == 400 and "reasoning" in r.text:
            body.pop("reasoning_effort", None)          # provider doesn't support it
            r = await _http().post("/chat/completions", json=body)
    if r.status_code >= 400:
        raise LLMError(f"{r.status_code} from LLM ({model}): {_error_text(r)}")
    content = r.json()["choices"][0]["message"].get("content") or ""
    if not content.strip():
        raise LLMError(f"empty reply from {model} (token budget used up by reasoning?)")
    return parse_json(content)


def _error_text(r: httpx.Response) -> str:
    try:
        return r.json().get("error", {}).get("message", r.text)[:200]
    except Exception:
        return r.text[:200]


async def resolve_models() -> None:
    """Check configured LLM and STT models exist; swap in a working one if retired."""
    try:
        r = await _http().get("/models")
        r.raise_for_status()
        available = {m["id"] for m in r.json().get("data", [])}
    except Exception as e:
        log.warning("Could not list models at %s (%s); using configured names as-is",
                    config.LLM_BASE_URL, e)
        return
    for attr, fallbacks, same_provider in (
            ("LLM_MODEL", FALLBACK_MODELS, True),
            ("LLM_SUMMARY_MODEL", FALLBACK_MODELS, True),
            ("LLM_CHECK_MODEL", FALLBACK_MODELS, True),
            ("STT_API_MODEL", FALLBACK_STT, config.STT_API_BASE_URL == config.LLM_BASE_URL)):
        current = getattr(config, attr)
        if not same_provider or not current or current in available:
            continue
        pick = next((m for m in fallbacks if m in available), None)
        if pick:
            log.warning("%s '%s' is not offered by the provider (retired?) – using '%s'. "
                        "Set %s in .env to choose another.", attr, current, pick, attr)
            setattr(config, attr, pick)
        else:
            log.error("%s '%s' not available. Available: %s", attr, current, sorted(available))


def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


# ---------------------------------------------------------------- live cues
def person_brief(person) -> str:
    if person is None:
        return "Unknown person (first meeting)."
    p = person.profile or {}
    parts = [f"Name: {person.name if person.named else 'unknown'}"]
    for k in ("relationship", "designation", "key_facts", "interests", "communication_tips", "follow_ups"):
        if p.get(k):
            v = p[k] if isinstance(p[k], str) else "; ".join(p[k])
            parts.append(f"{k}: {v}")
    if getattr(person, "notes", ""):
        parts.append(f"Wearer's own notes: {person.notes}")
    if person.latest_summary:
        parts.append(f"Last conversation: {person.latest_summary}")
    return "\n".join(parts)


def format_transcript(turns: list[dict]) -> str:
    out = []
    for t in turns:
        who = "ME" if t["speaker"] == "me" else "THEM"
        flag = " [unclear]" if t.get("conf", 1) < 0.45 else ""
        out.append(f"{who}: {t['text']}{flag}")
    return "\n".join(out) or "(nothing yet)"


# Sayable lines that state no new fact. Used when a written reply invented one,
# or the suggester returned nothing rather than invent one. Order is stable.
_GENERIC_REPLIES = {
    "question": [
        "Good question, let me think",
        "What do you think?",
        "Hmm, what about you?",
        "Can you say a bit more?",
    ],
    "statement": [
        "Tell me more about that",
        "How are you feeling about it?",
        "That sounds interesting",
        "What happened next?",
    ],
    "opener": [
        "Hey, good to see you",
        "Hey, how's it going?",
    ],
    "icebreaker": [
        "How's your week been?",
        "What have you been up to?",
        "Got any plans later?",
    ],
}


def _word_count(text: str) -> int:
    return len(re.findall(r"[a-z0-9']+", (text or "").lower()))


def _reply_kind(moment: str, last_them: str) -> str:
    kind = (moment or "").strip().lower()
    if kind in ("opener", "icebreaker"):
        return kind
    if "asked a question" in (moment or "").lower() or "?" in (last_them or ""):
        return "question"
    return "statement"


def _generic_choices(moment: str, last_them: str, max_words: int) -> list[str]:
    pool = _GENERIC_REPLIES[_reply_kind(moment, last_them)]
    limit = max(int(max_words or 0), 1)
    fitting = [g for g in pool if _word_count(g) <= limit]
    return fitting or list(pool)


def _ensure_replies(raw: list[str], kept: list[str], *, fill_empty: int, moment: str,
                    max_words: int, last_them: str = "") -> list[str]:
    """Keep verified lines. A dropped or missing line becomes a generic one.

    Survivors stay in their original slots. Generics fill only what the checker
    removed, or the whole list when the suggester refused to write anything.
    An honest short list is not padded up to the quota.
    """
    choices = _generic_choices(moment, last_them, max_words)
    seen = {line.lower() for line in kept}

    def take() -> str | None:
        for g in choices:
            key = g.lower()
            if key not in seen:
                seen.add(key)
                return g
        return None

    if not raw:
        out = []
        for _ in range(max(0, fill_empty)):
            g = take()
            if not g:
                break
            out.append(g)
        if out:
            log.info("suggester returned nothing; showing generic replies: %s", out)
        return out

    out = []
    k = 0
    replaced = 0
    for line in raw:
        if k < len(kept) and kept[k] == line:
            out.append(line)
            k += 1
            continue
        g = take()
        if g:
            out.append(g)
            replaced += 1
    if replaced:
        log.info("replaced %d hallucinated replies with generic lines", replaced)
    return out


async def live_cues(person, turns: list[dict], settings: dict, moment: str = "") -> dict:
    """Returns {"replies": [up to n], "social": str|None, "emotion": str|None}.

    The caller has already decided this moment is worth a suggestion. The suggester
    is not asked again to fill a quota. A second model reviews the lines and drops
    any that invent a fact. A dropped line, or an empty list, is replaced with a
    generic reaction or an open question so the wearer still has something to say.
    """
    n = max(0, min(int(settings.get("num_replies", 2)), 4))
    want_social = settings.get("social", True)
    want_emotion = settings.get("emotion", True)
    max_words = int(settings.get("max_words", 10))

    schema = {}
    if n:
        schema["replies"] = (
            f"array of 1 to {n} different short replies the wearer can say. "
            "Reactions, empathy, follow-ups, and paraphrases are welcome, even in new words. "
            "Never include a fact, name, place, time, job, or plan that is not in the transcript "
            "or profile, and never answer the wearer's own life for them. "
            "If a specific reply would invent a fact, use a generic reaction or an open question")
    if want_social:
        schema["social"] = ("one short social-cue tip (e.g. 'They asked a question - answer, then ask back') "
                            "or null if none is worth showing")
    if want_emotion:
        schema["emotion"] = ("THEM's likely feeling in 1-3 words, from their words only "
                             "(e.g. 'excited', 'a bit worried') or null if it is not clear")
    if not schema:
        return {"replies": [], "social": None, "emotion": None}

    last_them = next((t["text"] for t in reversed(turns) if t["speaker"] == "them"), "")
    user = prompts.CUE_USER.format(
        person=person_brief(person),
        transcript=format_transcript(turns[-config.CUE_CONTEXT_TURNS:]),
        last_them=last_them,
        moment=moment or "A reply is needed. Use a generic reaction if a specific one would invent a fact.",
        schema=json.dumps(schema, indent=1))
    system = prompts.CUE_SYSTEM.format(max_words=max_words)

    data = await chat_json(system, user, max_tokens=180)
    replies = _clean_list(data.get("replies"), max_words)
    # De-dupe while keeping order. Do not call the model again to pad the list.
    seen: set[str] = set()
    unique = []
    for r in replies:
        if r.lower() not in seen:
            seen.add(r.lower())
            unique.append(r)
    raw = unique[:n]
    checked = await filter_hallucinations(_cue_context(person, turns), raw)
    if n:
        checked = _ensure_replies(
            raw, checked, fill_empty=n, moment=moment, max_words=max_words, last_them=last_them)
    return {
        "replies": checked,
        "social": _clip(data.get("social"), max_words + 4) if want_social else None,
        "emotion": _clip(data.get("emotion"), 4) if want_emotion else None,
    }


def _cue_context(person, turns: list[dict]) -> str:
    """The transcript and profile the suggester saw, for the hallucination check."""
    transcript = format_transcript(turns[-config.CUE_CONTEXT_TURNS:])
    return f"Known about this person:\n{person_brief(person)}\n\nConversation so far:\n{transcript}"


def _keep_indexes(keep, n: int) -> set[int]:
    """Indexes the checker asked to show. Ignores anything that is not a real index."""
    chosen: set[int] = set()
    for item in keep:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            idx = item
        elif isinstance(item, str) and item.strip().lstrip("-").isdigit():
            idx = int(item.strip())
        else:
            continue
        if 0 <= idx < n:
            chosen.add(idx)
    return chosen


async def filter_hallucinations(context: str, replies: list[str]) -> list[str]:
    """Drop replies that invent a concrete fact. Keep relevant help in new wording.

    An empty list skips the checker. If the checker fails or omits `keep`, every
    reply is dropped so an unverified line is never shown. The caller then puts
    a generic line in each empty slot. Order of the survivors stays as written.
    """
    if not replies:
        return []
    numbered = "\n".join(f"{i}. {reply}" for i, reply in enumerate(replies))
    user = prompts.CHECK_USER.format(context=context, replies=numbered)
    try:
        data = await chat_json(
            prompts.CHECK_SYSTEM, user,
            model=config.LLM_CHECK_MODEL or None,
            max_tokens=80, temperature=0)
    except Exception as e:
        log.warning("hallucination check failed: %s", e)
        return []
    keep = data.get("keep") if isinstance(data, dict) else None
    if not isinstance(keep, list):
        log.warning("hallucination check returned no keep list")
        return []
    chosen = _keep_indexes(keep, len(replies))
    dropped = [replies[i] for i in range(len(replies)) if i not in chosen]
    if dropped:
        log.info("checker dropped %s", dropped)
    return [reply for i, reply in enumerate(replies) if i in chosen]


def worth_opener(person) -> bool:
    """True when memory has something a conversation could start from.

    Key facts alone are not enough. An allergy or a pet name is not an opener.
    """
    if person is None:
        return False
    prof = person.profile or {}
    if any(str(x).strip() for x in (prof.get("follow_ups") or [])):
        return True
    if int(getattr(person, "conversation_count", 0) or 0) > 0:
        return True
    return bool((getattr(person, "latest_summary", "") or "").strip())


def opener_hook(person) -> str:
    """The memory an opener should lead with, or a note that a hello is enough."""
    prof = (person.profile or {}) if person else {}
    follows = [str(x).strip() for x in (prof.get("follow_ups") or []) if str(x).strip()]
    if follows:
        return follows[0]
    summary = (getattr(person, "latest_summary", "") or "").strip()
    if summary:
        return ("Nothing listed as a follow-up. If the last conversation left something unfinished, "
                "ask about that. Otherwise a warm hello.")
    return "Nothing specific. A warm hello is enough."


async def opening_line(person, settings: dict) -> list[str]:
    """One sayable starter from memory, or nothing. Does not call the model without memory."""
    if not worth_opener(person):
        return []
    # A greeting plus a question needs a little more room than a mid-chat reply.
    max_words = max(int(settings.get("max_words", 10)), 14)
    user = prompts.OPENING_USER.format(person=person_brief(person), hook=opener_hook(person))
    system = prompts.OPENING_SYSTEM.format(max_words=max_words)
    data = await chat_json(system, user, max_tokens=80, temperature=0.3)
    lines = _clean_list(data.get("replies"), max_words)[:1]
    context = (
        f"Known about this person:\n{person_brief(person)}\n\n"
        f"Bring up this first: {opener_hook(person)}\n\n"
        "Nobody has spoken yet."
    )
    kept = await filter_hallucinations(context, lines)
    # A rejected or missing starter still becomes a hello. The moment was worth opening.
    return _ensure_replies(lines, kept, fill_empty=1, moment="opener", max_words=max_words)


async def icebreaker(person, turns: list[dict], settings: dict) -> dict:
    """A new easy thread when the conversation has stalled. Not a reply to the last line.

    One model writes the lines. The checker drops any that invent a fact. An empty
    or rejected list becomes a generic question, so the wearer still has a way back in.
    """
    if int(settings.get("num_replies", 1) or 0) <= 0:
        return {"replies": [], "social": None, "emotion": None, "kind": "icebreaker"}
    max_words = int(settings.get("max_words", 10))
    user = prompts.ICEBREAKER_USER.format(
        person=person_brief(person),
        transcript=format_transcript(turns[-config.CUE_CONTEXT_TURNS:]))
    system = prompts.ICEBREAKER_SYSTEM.format(max_words=max_words)
    data = await chat_json(system, user, max_tokens=120, temperature=0.4)
    replies = _clean_list(data.get("replies"), max_words)
    seen: set[str] = set()
    unique = []
    for r in replies:
        if r.lower() not in seen:
            seen.add(r.lower())
            unique.append(r)
    raw = unique[:1]
    context = (
        f"Known about this person:\n{person_brief(person)}\n\n"
        f"Conversation so far:\n{format_transcript(turns[-config.CUE_CONTEXT_TURNS:])}\n\n"
        "These lines are ice breakers. They open a new thread. They are not answers."
    )
    kept = await filter_hallucinations(context, raw)
    lines = _ensure_replies(
        raw, kept, fill_empty=1, moment="icebreaker", max_words=max_words)[:1]
    social = "Try a new topic" if settings.get("social", True) else None
    return {"replies": lines, "social": social, "emotion": None, "kind": "icebreaker"}


def _clean_list(v, max_words) -> list[str]:
    if isinstance(v, str):
        v = [v]
    return [c for c in (_clip(x, max_words) for x in (v or [])) if c]


def _clip(s, max_words) -> str | None:
    """Keep the whole line. The prompt asks for a short caption; we do not chop it
    or stamp an ellipsis on the end."""
    del max_words
    if not s or not isinstance(s, str) or s.strip().lower() in ("null", "none", "n/a"):
        return None
    return s.strip().strip('"')


# ---------------------------------------------------------------- end of conversation
async def summarize(person, turns: list[dict]) -> dict:
    """-> {"record": {title, summary, topics, mood, key_points, follow_ups}, "name", "profile"}"""
    user = prompts.SUMMARY_USER.format(
        profile=json.dumps(person.profile if person else {}, ensure_ascii=False),
        latest=(person.latest_summary if person else ""),
        notes=(getattr(person, "notes", "") if person else "") or "(none)",
        transcript=format_transcript(turns))
    data = await chat_json(prompts.SUMMARY_SYSTEM, user,
                           model=config.LLM_SUMMARY_MODEL or None,
                           max_tokens=900, temperature=0.2)
    profile = data.get("profile") or {}
    for k in ("key_facts", "interests", "communication_tips", "follow_ups"):
        profile[k] = _str_list(profile.get(k), 8)
    for k in ("relationship", "recall_hint", "designation"):
        profile[k] = str(profile.get(k) or "").strip()
    if person:
        for k in ("relationship", "designation"):
            if person.profile.get(k) and not profile[k]:
                profile[k] = str(person.profile[k]).strip()
    profile["designation"] = profile["designation"][:60]
    record = {
        "title": str(data.get("title") or "").strip()[:80],
        "summary": str(data.get("summary") or "").strip(),
        "topics": _str_list(data.get("topics"), 5),
        "mood": str(data.get("mood") or "").strip()[:40],
        "key_points": _str_list(data.get("key_points"), 5),
        "follow_ups": _str_list(data.get("conversation_follow_ups"), 5),
    }
    name = data.get("name")
    return {"record": record, "name": name if isinstance(name, str) and name.strip() else None,
            "profile": profile}


def _str_list(v, limit: int) -> list[str]:
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        return []
    return [str(x).strip() for x in v if str(x).strip()][:limit]
