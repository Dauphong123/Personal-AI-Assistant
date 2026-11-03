"""Language helpers shared by the text chat and voice mode (English and Vietnamese).

Vietnamese is recognized from its letters: the Latin Extended Additional block
(U+1EA0 to U+1EF9: ạ ả ấ ầ ể ệ ớ ự ...) is used only by Vietnamese, as are ă, đ, ơ and ư.
Text typed without accents ("xin chao") cannot be told apart from English and counts as English.
"""

import unicodedata

VIETNAMESE_LETTERS = set("ăĂđĐơƠưƯ")
# Vietnamese has many tone-marked letters (a quarter of all letters or more in a sentence); an
# English sentence that quotes one name ("Nguyễn") has far fewer.
MIN_VIETNAMESE_SHARE = 0.10


def normalize(text: str) -> str:
    """Composed form (NFC): a keyboard or recognizer may send letter + accent separately."""
    return unicodedata.normalize("NFC", text)


# Accented letters Vietnamese shares with French, Portuguese and others (à á â ã è é ê ì í ò ó ô õ ù
# ú ý). They count for half: they point to Vietnamese, but "café" alone must not.
SHARED_ACCENTED_LETTERS = set("àáâãèéêìíòóôõùúýÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝ")


def _vietnamese_weight(char: str) -> float:
    if char in VIETNAMESE_LETTERS or "Ạ" <= char <= "ỹ":
        return 1.0  # used by Vietnamese only

    return 0.5 if char in SHARED_ACCENTED_LETTERS else 0.0


def detect_language(text: str) -> str:
    """'vi' if the text looks Vietnamese, otherwise 'en'."""
    letters = [c for c in normalize(text) if c.isalpha()]

    if not letters:
        return "en"

    score = sum(_vietnamese_weight(c) for c in letters)
    return "vi" if score / len(letters) >= MIN_VIETNAMESE_SHARE else "en"


STRINGS = {
    "en": {
        "apology": "Sorry, something went wrong while handling that.",
        "approve_intro": "I want to run",
        "approve_join": "; and ",
        "approve_outro": "Say yes, allow it, to approve.",
        "truncated": "The rest is on screen.",
    },
    "vi": {
        "apology": "Xin lỗi, đã có lỗi khi xử lý yêu cầu đó.",
        "approve_intro": "Tôi muốn chạy",
        "approve_join": "; và ",
        "approve_outro": "Hãy nói đồng ý để cho phép.",
        "truncated": "Phần còn lại hiển thị trên màn hình.",
    },
}


def say(key: str, language: str = "en") -> str:
    """A localized fixed phrase (falls back to English)."""
    return STRINGS.get(language, STRINGS["en"])[key]
