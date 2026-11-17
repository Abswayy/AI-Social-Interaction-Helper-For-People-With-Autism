"""When a live reply suggestion is worth asking for.

Suggestions used to run after every line from the other person. Short lines,
backchannels, and half-finished clauses then forced the model to invent a
reply. A reply is requested only for a real prompt: a question, or a finished
statement. The session waits until the wearer has been quiet for a few seconds,
or has hesitated a couple of times, and later offers an ice breaker if the
talk still has not moved.

This module decides whether a line is worth answering, and whether the wearer
sounds stuck. The session owns the clock.
"""
from __future__ import annotations

import re

from . import config

# Whole line is just acknowledgement, not something to answer.
_FILLER_PHRASES = {
    "yeah", "yea", "yep", "yup", "yes", "no", "nah", "ok", "okay", "k",
    "hmm", "hm", "uh", "um", "uh huh", "uh-huh", "mhm", "mm", "mmm",
    "right", "sure", "cool", "nice", "wow", "oh", "ah", "aha", "alright",
    "all right", "got it", "i see", "thanks", "thank you", "lol", "haha",
    "hahaha", "okay okay", "yeah yeah", "oh wow", "oh ok", "oh okay",
}
_FILLER_WORDS = {
    "yeah", "yea", "yep", "yup", "yes", "no", "nah", "ok", "okay", "k",
    "hmm", "hm", "uh", "um", "uh-huh", "mhm", "mm", "mmm", "right", "sure",
    "cool", "nice", "wow", "oh", "ah", "aha", "alright", "thanks", "thank",
    "lol", "haha", "hahaha",
}
# Last word means the clause is still open ("we went to the", "and then").
_DANGLING = {
    "and", "but", "so", "because", "or", "like", "then", "if", "that",
    "the", "a", "an", "to", "of", "with", "for",
}
_Q_PREFIXES = (
    "what ", "why ", "how ", "when ", "where ", "who ", "which ",
    "what's ", "whats ", "how's ", "hows ", "who's ", "whos ",
    "do you ", "did you ", "can you ", "could you ", "would you ", "will you ",
    "are you ", "have you ", "tell me ", "is ", "isn't ", "aren't ",
    "was ", "were ", "do ", "did ", "does ", "don't ", "can ", "could ",
    "would ", "will ", "have ", "has ", "should ",
)

MOMENT_TEXT = {
    "question": "They asked a question and then paused. A short answer is appropriate.",
    "statement": "They finished a statement and left a pause. "
                 "Suggest a short reaction or a follow-up. "
                 "If a specific reply would invent a fact, use a generic reaction or an open question.",
}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _is_filler(words: list[str]) -> bool:
    phrase = " ".join(words)
    if phrase in _FILLER_PHRASES:
        return True
    return len(words) <= 4 and all(w in _FILLER_WORDS for w in words)


def _is_question(text: str, words: list[str]) -> bool:
    if "?" in (text or ""):
        return True
    phrase = " ".join(words) + " "
    return phrase.startswith(_Q_PREFIXES)


def cue_decision(turn: dict | None) -> tuple[str | None, str]:
    """Return (moment, why). moment is 'question' or 'statement', or None to stay quiet."""
    if not turn or turn.get("speaker") != "them":
        return None, "not the other person"
    text = (turn.get("text") or "").strip()
    words = _words(text)
    if not words:
        return None, "empty"
    if float(turn.get("conf") or 0) < config.CUE_MIN_CONF:
        return None, "low confidence"
    if _is_noise(text, words):
        return None, "noise"
    if _is_filler(words):
        return None, "backchannel"
    if words[-1] in _DANGLING and "?" not in text:
        return None, "unfinished"
    if _is_question(text, words):
        if len(words) < config.CUE_MIN_QUESTION_WORDS:
            return None, "question too short"
        return "question", "they asked something"
    if len(words) < config.CUE_MIN_WORDS:
        return None, "too short to answer"
    return "statement", "they finished a thought"


def cue_moment(turn: dict | None) -> str | None:
    return cue_decision(turn)[0]


# Sounds and short phrases that mean the wearer is stuck, not that they answered.
# Whisper spells the same stall many ways: "um", "uhm", "uhh", "eh", "hum", "erm".
_HESITATION_WORD = re.compile(
    r"^(?:"
    r"u+h*m+h*"    # um, umm, uhm, umh
    r"|u+h+"       # uh, uhh
    r"|a+h+"       # ah, ahh
    r"|a{2,}"      # aa — a drawn-out stall, not the word "a"
    r"|e+h*m+"     # em, ehm
    r"|e+h+"       # eh, ehh
    r"|e+r+m*"     # er, err, erm
    r"|h+u+m+"     # hum
    r"|h+m+"       # hm, hmm
    r"|m+"         # mm, mmm
    r"|huh+"
    r")$"
)
_HESITATION_PHRASES = {
    "i don't know", "i dont know", "i do not know", "i dunno", "dunno",
    "not sure", "im not sure", "i'm not sure", "not really sure",
    "i'm unsure", "im unsure", "no idea",
    "let me think", "hang on", "hold on", "hold on a sec",
    "one sec", "one second", "give me a sec", "come again",
    "pardon", "sorry what", "wait",
}
# Words that can follow "um" / "ah" and still be a stall, not an answer.
_HESITATION_TAIL = {"so", "like", "wait", "well", "okay", "ok", "sorry", "just", "and"}
# Agreement, not confusion. "uh huh" is two words after punctuation is stripped.
_NOT_HESITATION = {
    "uh huh", "uh-huh", "uh uh", "mm hmm", "mm hm", "mmhmm", "mhm", "mhmm", "mmhm",
}


def _phrase_is_hesitation(phrase: str) -> bool:
    if phrase in _HESITATION_PHRASES:
        return True
    return any(phrase == p or phrase.startswith(p + " ") for p in _HESITATION_PHRASES)


def wearer_hesitating(text: str) -> bool:
    """True when the wearer sounds unsure: "umm", "uh", "ah...", "I don't know".

    A short stall in front still counts ("um, I don't know", "uh so").
    Agreement ("uh huh", "mhm") does not.
    """
    words = _words(text)
    if not words:
        return False
    phrase = " ".join(words)
    if phrase in _NOT_HESITATION:
        return False
    if _phrase_is_hesitation(phrase):
        return True
    if all(_HESITATION_WORD.match(w) for w in words):
        return True
    lead = 0
    while lead < len(words) and _HESITATION_WORD.match(words[lead]):
        lead += 1
    if lead == 0:
        return False
    rest = words[lead:]
    rest_phrase = " ".join(rest)
    if rest_phrase in _NOT_HESITATION:
        return False
    if _phrase_is_hesitation(rest_phrase):
        return True
    # "um so", "ah wait" — a hesitation with almost nothing after it.
    return len(rest) <= 2 and all(
        w in _HESITATION_TAIL or _HESITATION_WORD.match(w) for w in rest)


def hesitation_hits(text: str) -> int:
    """How many stall sounds are in this line. 0 when it is not a hesitation.

    One "umm" is 1. "umm ah" is 2. A stuck phrase ("I don't know") counts as a
    few on its own, so the wearer does not have to umm again.
    """
    if not wearer_hesitating(text):
        return 0
    words = _words(text)
    fillers = sum(1 for w in words if _HESITATION_WORD.match(w))
    lead = 0
    while lead < len(words) and _HESITATION_WORD.match(words[lead]):
        lead += 1
    rest = " ".join(words[lead:])
    phrase = " ".join(words)
    if _phrase_is_hesitation(phrase) or (rest and _phrase_is_hesitation(rest)):
        return max(fillers, 2)
    return max(fillers, 1)


# Coughs and throat noises. Whisper often writes them as words, and a long
# enough cough is a real utterance, not the short blip the VAD throws away.
_NOISE_WORD = re.compile(
    r"^(?:cough(?:ing|ed|s)?|ahem+|achoo+|sneeze(?:d|s|ing)?|sniff(?:le|ing|ed|s)?|"
    r"khm+|hem+|ugh+|clearing|throat)$"
)


def _is_noise(text: str, words: list[str] | None = None) -> bool:
    raw = (text or "").strip().lower()
    if not raw:
        return False
    if re.fullmatch(
            r"[\(\[\*<\"']?\s*(?:cough\w*|ahem+|achoo+|sneeze\w*|sniff\w*|khm+|hem+)\s*[\)\]\*>\"'.!?]*",
            raw):
        return True
    words = _words(raw) if words is None else words
    if not words:
        return bool(re.search(r"cough|ahem|achoo|sneeze|sniff", raw))
    if any(_NOISE_WORD.match(w) for w in words) and len(words) <= 4:
        return True
    return False


def is_ignorable(text: str) -> bool:
    """Cough, backchannel, or stall. Not something the glasses should react to."""
    words = _words(text)
    if not words:
        return True
    return _is_noise(text, words) or _is_filler(words) or wearer_hesitating(text)


def needs_repeat(turn: dict | None) -> bool:
    """A substantial line we could not hear well enough to answer."""
    if not turn or turn.get("speaker") != "them":
        return False
    if float(turn.get("conf") or 0) >= config.CUE_REPEAT_BELOW:
        return False
    if cue_moment(turn):
        return False
    return len(_words(turn.get("text") or "")) >= config.CUE_MIN_WORDS


# Words that don't tell us whether the wearer echoed a suggestion.
_STOP = {
    "i", "me", "my", "you", "your", "we", "our", "it", "its", "a", "an", "the",
    "to", "of", "and", "or", "is", "am", "are", "was", "were", "be", "do", "did",
    "so", "just", "that", "this", "im", "i'm", "it's", "its",
}


def _content_words(text: str) -> set[str]:
    return {w for w in _words(text) if w not in _STOP and len(w) > 2}


def used_suggestion(said: str, replies: list[str]) -> bool:
    """True when what was just said echoes one of the lines on screen."""
    said_w = _content_words(said)
    if not said_w:
        return False
    for reply in replies or []:
        sug = _content_words(reply)
        overlap = {w for w in (said_w & sug) if len(w) >= 4}
        if len(overlap) >= 2:
            return True
        # A short reply ("nice", "where from") still counts when it repeats the suggestion.
        if len(overlap) >= 1 and len(_words(said)) <= 6:
            return True
    return False


# Words that don't tell us which subject a line is about.
_TOPIC_STOP = _STOP | {
    "just", "really", "very", "also", "too", "then", "about", "from", "with",
    "for", "have", "has", "had", "been", "being", "they", "them", "their",
    "what", "when", "where", "who", "why", "how", "which", "on", "in", "at",
    "actually", "here", "there",
}
# "he helped with the boxes" is still the move, not a new subject.
_REFERRING = {"he", "she", "they", "him", "her", "them", "his", "their"}
_CONTINUE = re.compile(
    r"^(?:and|also|plus|because|i mean|like i|yeah and|and then|but yeah|so yeah)\b"
)


def _stem(word: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(word) > len(suf) + 3 and word.endswith(suf):
            return word[:-len(suf)]
    return word


def _topic_words(text: str) -> set[str]:
    return {_stem(w) for w in _words(text) if w not in _TOPIC_STOP and len(w) > 2}


def _topic_overlap(left: set[str], right: set[str]) -> bool:
    if left & right:
        return True
    for a in left:
        for b in right:
            if len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a)):
                return True
    return False


def topic_changed(text: str, anchor: str = "", replies: list[str] | None = None) -> bool:
    """True when `text` leaves the subject the suggestion was about.

    A cough, a backchannel, a stall, or another line on the same subject is
    not a change. A new question, or a full remark that shares nothing with
    the line we answered or the lines on screen, is.
    """
    words = _words(text)
    if not words or _is_filler(words) or _is_noise(text, words) or wearer_hesitating(text):
        return False
    if _CONTINUE.match(" ".join(words)):
        return False
    bag: set[str] = set()
    for src in (anchor, *(replies or [])):
        bag |= _topic_words(src)
    mine = _topic_words(text)
    if mine and _topic_overlap(mine, bag):
        return False
    if _is_question(text, words) and len(mine) >= 2:
        return True
    if not _is_question(text, words) and any(w in _REFERRING for w in words):
        return False
    return len(words) >= 6 and len(mine) >= 3


def suggestion_expired(turn: dict | None, replies: list[str] | None = None, anchor: str = "",
                       *, wearer: bool = False) -> bool:
    """Take the on-screen lines down when they are no longer what to say.

    They stay up through a cough, a backchannel, a stall, and the other person
    adding to the same subject. They leave when the wearer says one, when the
    wearer answers in their own words, or when the subject changes.
    `anchor` is the line the suggestion was answering.
    """
    if not turn:
        return False
    text = turn.get("text") or ""
    if used_suggestion(text, replies or []):
        return True
    words = _words(text)
    # A short noise can be transcribed as a sentence. It is not a new subject.
    voiced = turn.get("voiced_ms")
    short = voiced is not None and int(voiced) < 700 and not _is_question(text, words)
    # The wearer already answered, just not with the line we offered.
    # A cough or an "umm" is not an answer, so the lines stay.
    if wearer and not is_ignorable(text):
        if short and _is_noise(text, words):
            return False
        return True
    if short:
        return False
    return topic_changed(text, anchor=anchor, replies=replies or [])
