"""Dictation cleanup: turn what the recognizer heard into the prompt you meant (as Wispr Flow does).

Rule-based on purpose: an LLM pass would add seconds per prompt at ~4 tok/s. Only explicit,
punctuated cues are acted on, so ordinary speech passes through unchanged:
- hesitations (um, uh, ờ, ừm) go; Vietnamese particles with meaning ("à", "ừ") stay;
- stutters ("the the") collapse, except grammatical repeats and Vietnamese reduplication;
- ", no wait, Y" / ", I mean, Y" / ", à không, Y" replace what Y corrects;
- "scratch that" / "xóa cái đó" drops the sentence before it;
- first letter capitalized, final period added.
"""

import re

from language import detect_language, normalize

FILLER = r"(?:u+m+|u+h+|uhm+|e+r+m*|h+m+|m{2,}|ờ+|ừm+|ơ+)"
FILLER_RE = re.compile(rf",?\s*(?<!\w){FILLER}(?!\w)[\s,.…]*", re.IGNORECASE)

SCRATCH_RE = re.compile(
    r"[\s,;]*\b(?:scratch that|delete that|forget that|xóa cái đó|xoá cái đó|bỏ câu đó)\b[\s,.!?;]*",
    re.IGNORECASE,
)
# The cue must follow a pause mark, so "I mean it" in the middle of a sentence is left alone.
CORRECTION_RE = re.compile(
    r"\s*(?:,|;|—|–|-|\.\.\.|…)\s*"
    r"(?:no,?\s+wait|wait,?\s+no|no,?\s+no|i mean|oops|à không|à nhầm|nhầm rồi|ý tôi là)"
    r"\b[\s,:]*",
    re.IGNORECASE,
)
SENTENCE_END = re.compile(r"[.!?…]+\s+")
MAX_REPLACED_WORDS = 3  # a short unanchored correction replaces this many words at most

# Repeats that are grammar, not stutter ("she said that that was fine", "had had").
KEEP_REPEATED = {"that", "had"}
STUTTER_RE = re.compile(r"\b([^\W\d_]+)(?:\s+\1\b)+", re.IGNORECASE)


def _sentence_start(text: str) -> int:
    """Index where the last sentence of `text` begins."""
    ends = list(SENTENCE_END.finditer(text))
    return ends[-1].end() if ends else 0


def _word(token: str) -> str:
    return re.sub(r"[^\w]", "", token).lower()


def _scratch(text: str) -> str:
    while match := SCRATCH_RE.search(text):
        before = text[: match.start()].rstrip(" ,;.!?…")
        text = before[: _sentence_start(before)] + text[match.end() :]

    return text


def _correct(text: str) -> str:
    while match := CORRECTION_RE.search(text):
        prefix, suffix = text[: match.start()], text[match.end() :]
        start = _sentence_start(prefix)
        wrong = prefix[start:].split()
        end = SENTENCE_END.search(suffix)
        right_text = suffix[: end.start()] if end else suffix
        right = right_text.split()

        if not wrong or not right:
            text = prefix + (" " if right else "") + suffix
            continue

        first = _word(right[0])
        anchors = [i for i, token in enumerate(wrong) if _word(token) == first]

        if anchors:  # "open the file, I mean the config file": restart from "the"
            kept = wrong[: anchors[-1]]
        elif len(right) <= MAX_REPLACED_WORDS and len(right) < len(wrong) + 1:
            kept = wrong[: max(0, len(wrong) - len(right))]  # "2 pm, no wait, 3 pm"
        else:  # nothing says what is being corrected: keep both parts
            text = prefix + ", " + suffix
            continue

        text = prefix[:start] + " ".join(kept + [right_text.strip()]) + suffix[len(right_text) :]

    return text


def _unstutter(text: str) -> str:
    if detect_language(text) == "vi":  # "từ từ", "đi đi": reduplication carries meaning
        return text

    def collapse(match: re.Match) -> str:
        return match.group(0) if match.group(1).lower() in KEEP_REPEATED else match.group(1)

    return STUTTER_RE.sub(collapse, text)


def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:…])", r"\1", text)
    text = re.sub(r",(?:\s*,)+", ",", text)
    text = re.sub(r"^[\s,.;:!?…-]+", "", text)
    text = text.rstrip(" ,;:-")

    if not any(ch.isalnum() for ch in text):
        return ""

    text = text[0].upper() + text[1:]

    if text[-1] not in ".!?…":
        text += "."

    return text


def clean_dictation(text: str) -> str:
    """The prompt you meant to say; '' when nothing is left."""
    text = normalize(text or "").strip()
    text = FILLER_RE.sub(" ", text)
    text = _scratch(text)
    text = _correct(text)
    text = _unstutter(text)
    return _tidy(text)
