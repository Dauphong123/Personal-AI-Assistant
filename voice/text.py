"""Text helpers around speech: cleaning transcripts and replies."""

import re

from language import detect_language, normalize, say

YES_WORDS = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "approve", "approved", "allow", "go ahead", "do it", "confirm",
    # Vietnamese: yes / agreed / allowed / alright / do it
    "có", "vâng", "dạ", "ừ", "đồng ý", "được", "được rồi", "cho phép", "làm đi", "chạy đi",
}  # fmt: skip
YES_FIRST_WORDS = {"yes", "yeah", "yep", "có", "vâng", "dạ"}
YES_PREFIXES = ("đồng ý", "cho phép", "được rồi", "làm đi", "chạy đi")
# An approval runs a command or changes files, so doubt means no. An answer with a second
# thought in it ("yes, no wait, don't", "có... không, đừng") or a long ramble is declined.
MAX_ANSWER_WORDS = 8
DOUBT_WORDS = frozenset(
    "no nope nah not don dont wait hold never cancel stop maybe unsure but however "
    "không khong đừng khoan chưa thôi nhưng".split()
)


# What Whisper tends to "hear" in noise or in the clipped tail of a wake word.
FILLER = {
    "you", "uh", "um", "umm", "hmm", "mm", "ah", "oh", "huh", "the", "a", "bye", "thank you", "thanks for watching",
    # Vietnamese: hesitations, thanks, goodbye
    "ờ", "ừm", "à", "ơ", "ừ", "cảm ơn", "cảm ơn các bạn", "cảm ơn bạn đã xem", "hẹn gặp lại", "tạm biệt",
}  # fmt: skip
# Whisper's best-known Vietnamese hallucination on silence: a YouTube "subscribe" outro.
FILLER_CONTAINS = ("subscribe cho kênh", "ghiền mì gõ", "đăng ký kênh", "like và subscribe", "bỏ lỡ những video")


def is_filler(text: str) -> bool:
    """True for transcripts that are recognizer noise rather than a request."""
    cleaned = " ".join(re.sub(r"[^\w\s]", " ", normalize(text).lower()).split())
    return cleaned in FILLER or any(part in cleaned for part in FILLER_CONTAINS)


def wake_word_phrases(wake_word: str) -> list[str]:
    """Spoken forms of a wake word model name: 'hey_jarvis' -> ['hey jarvis', 'jarvis']."""
    name = wake_word.replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"\.onnx$", "", name, flags=re.I)
    name = re.sub(r"_v\d+(\.\d+)*$", "", name)  # model version suffix
    words = [w for w in re.split(r"[_\s-]+", name.lower()) if w]
    phrases = [" ".join(words)] if words else []

    if len(words) > 1 and words[0] in {"hey", "ok", "okay", "hi", "hello"}:
        phrases.append(" ".join(words[1:]))

    return phrases


def strip_wake_word(text: str, wake_word: str) -> str:
    """Remove a leading wake word from a transcript ('Jarvis, open it' -> 'open it')."""
    phrases = sorted(wake_word_phrases(wake_word), key=len, reverse=True)

    if not phrases:
        return text.strip()

    alternatives = "|".join(r"\s+".join(map(re.escape, p.split())) for p in phrases)
    # "Hey Jarvis", "Này Jarvis", "Ê Jarvis", "Jarvis ơi" ("ơi" is a Vietnamese calling particle)
    pattern = (
        rf"^\W*(?:(?:hey|ok|okay|hi|hello|này|ê|chào|xin\s+chào)\W+)?(?:{alternatives})\b"
        r"(?:\W+ơi\b)?[\s,.:;!?-]*"
    )
    return re.sub(pattern, "", normalize(text).strip(), count=1, flags=re.I).strip()


def normalize_answer(text: str) -> str:
    """Map a spoken answer to what ChatLoop understands: 'y', 'stop', or 'n'."""
    from turns import is_stop_request

    if is_stop_request(text):
        return "stop"

    cleaned = " ".join(re.sub(r"[^\w\s]", " ", normalize(text).lower()).split())

    words = cleaned.split()

    if len(words) > MAX_ANSWER_WORDS or any(word in DOUBT_WORDS for word in words):
        return "n"

    if cleaned in YES_WORDS or (words and words[0] in YES_FIRST_WORDS) or cleaned.startswith(YES_PREFIXES):
        return "y"

    return "n"


def clean_for_speech(text: str, max_chars: int) -> str:
    """Turn a model reply into something worth reading aloud."""
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<think>.*", " ", text, flags=re.S | re.I)  # unterminated
    text = re.sub(r"```.*?```", " (code omitted) ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", " link ", text)
    text = re.sub(r"^\s*[-=*_]{3,}\s*$", " ", text, flags=re.M)  # horizontal rules
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)
    text = re.sub(r"[*_~|]{1,3}", "", text)
    text = " ".join(text.split())

    if len(text) > max_chars:
        cut = text[:max_chars]
        end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
        kept = cut[: end + 1] if end > max_chars // 2 else cut
        text = f"{kept} {say('truncated', detect_language(kept))}"

    return text


def split_sentences(text: str) -> list[str]:
    """Split into sentence-sized pieces so speech can start before synthesis ends."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in (s.strip() for s in parts) if p and re.search(r"\w", p)]
