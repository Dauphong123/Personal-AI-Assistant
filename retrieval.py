"""Ranking helpers shared by the knowledge-base search and the web page reader.

Vector search finds text that *means* the same; keyword (BM25) search finds the exact
names, error messages and numbers that embeddings blur. Combining the two with
reciprocal rank fusion beats either alone.
"""

import math
import re
import unicodedata
from collections import Counter

_TOKEN = re.compile(r"\w+")

# Words that say nothing about the topic. A query like "what is the capital of France" would
# otherwise match every document that happens to contain "is" and "the". Stored folded
# (no accents), so Vietnamese "là", "và", "của" are listed as "la", "va", "cua".
STOPWORDS = frozenset(
    "a an and are as at be but by for from how i if in is it its me my of on or so that the "
    "their then there these they this to was we were what when where which who why will with "
    "you your do does did can could should would about into than not no yes "
    "la va cua co khong cho cac nhung mot trong duoc de nay nhu the nao gi ai dau khi thi ma "
    "bi o tai voi den tu ra vao se da dang cung hay hoac neu vi sao".split()
)
# A word found in more than this share of the passages cannot tell them apart.
MAX_DOCUMENT_SHARE = 0.5
MIN_DOCUMENTS_FOR_SHARE = 8


def fold(text: str) -> str:
    """Lower-case and strip accents, so "tệp" and "tep" (typed without accents) match."""
    text = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    return "".join(c for c in text if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(fold(text))


class BM25:
    """Okapi BM25 over a small in-memory list of texts."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._tokens = [tokenize(d) for d in documents]
        self._counts = [Counter(t) for t in self._tokens]
        self._lengths = [len(t) for t in self._tokens]
        self._average = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        frequency: Counter = Counter()

        for counts in self._counts:
            frequency.update(counts.keys())

        total = len(documents)
        self._common = (
            {t for t, n in frequency.items() if n / total > MAX_DOCUMENT_SHARE}
            if total >= MIN_DOCUMENTS_FOR_SHARE
            else set()
        )
        self._idf = {
            term: math.log(1 + (total - n + 0.5) / (n + 0.5)) for term, n in frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        terms = {t for t in tokenize(query) if t not in STOPWORDS and t not in self._common}
        result = []

        for counts, length in zip(self._counts, self._lengths):
            score = 0.0

            for term in terms:
                count = counts.get(term, 0)

                if not count:
                    continue

                norm = self.k1 * (1 - self.b + self.b * length / (self._average or 1.0))
                score += self._idf[term] * count * (self.k1 + 1) / (count + norm)

            result.append(score)

        return result

    def top(self, query: str, k: int) -> list[int]:
        """Indexes of the best matches, best first; documents with no matching word are left out."""
        scores = self.scores(query)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [i for i in ranked[:k] if scores[i] > 0]


def reciprocal_rank_fusion(
    rankings: list[list], k: int = 60, weights: list[float] | None = None
) -> list[tuple]:
    """Merge several best-first lists of ids. Items ranked high in any list rise to the top.

    `weights` (one per list, default all 1) make a list count for more; it also decides ties."""
    fused: dict = {}
    weights = weights or [1.0] * len(rankings)

    for ranking, weight in zip(rankings, weights):
        for position, item in enumerate(ranking):
            fused[item] = fused.get(item, 0.0) + weight / (k + position + 1)

    return sorted(fused.items(), key=lambda pair: -pair[1])


def split_passages(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Cut text into passages of about `size` characters, on paragraph and sentence breaks."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    passages: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current

        if current.strip():
            passages.append(current.strip())

        current = current[-overlap:] if overlap and len(current) > overlap else ""

    for paragraph in paragraphs:
        pieces = [paragraph]

        if len(paragraph) > size:  # a long paragraph: cut it at sentence ends
            pieces = re.split(r"(?<=[.!?])\s+", paragraph)

        for piece in pieces:
            while len(piece) > size:  # no sentence ends at all: cut hard
                if current:
                    flush()

                passages.append(piece[:size])
                piece = piece[size - overlap :]

            if current and len(current) + len(piece) + 1 > size:
                flush()

            current = f"{current} {piece}".strip()

    if current.strip():
        passages.append(current.strip())

    return passages


def best_passages(text: str, query: str, max_chars: int, size: int = 800) -> str:
    """The passages of `text` that best match `query`, in their original order."""
    passages = split_passages(text, size=size, overlap=0)

    if not passages:
        return ""

    bm25 = BM25(passages)
    chosen = bm25.top(query, len(passages))
    kept: list[int] = []
    used = 0

    for index in chosen:
        if used + len(passages[index]) > max_chars and kept:
            break

        kept.append(index)
        used += len(passages[index])

    if not kept:  # nothing matched: fall back to the start of the text
        kept = [0]

    pieces = []
    previous = None

    for index in sorted(kept):
        if previous is not None and index != previous + 1:
            pieces.append("[...]")

        pieces.append(passages[index])
        previous = index

    return "\n\n".join(pieces)[: max_chars + size]
