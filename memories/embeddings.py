"""Embeddings for long-term memory (the RAG documents keep their own English model).

Facts are stored in English, but questions often come in Vietnamese: with the English-only
bge-small, "con mèo của tôi tên gì?" never found "The user's cat is named Miu." The multilingual
e5 model puts both languages in one space (tests/test_memory_eval.py: 36/40 -> 40/40). It is
384-dimensional like bge-small, so the tables keep their schema; stored rows are re-embedded
once when the model changes (`MemoryStore.ensure_embedding_model`).
"""

MEMORY_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# e5 models are trained with a "query: " or "passage: " prefix. Memory compares facts with facts
# (duplicates, outdated facts) as well as with questions, a symmetric use, for which the model
# card says to put "query: " on both sides.
QUERY_PREFIX = "query: "


class PrefixedEmbeddings:
    """Any `embed_query` embedder, with a fixed prefix on every text."""

    def __init__(self, base, prefix: str = QUERY_PREFIX):
        self._base = base
        self._prefix = prefix

    def embed_query(self, text: str) -> list[float]:
        return self._base.embed_query(self._prefix + text)


def memory_embeddings() -> PrefixedEmbeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    # CPU: a short text embeds in milliseconds, and the 4 GB GPU is already short for qwen3:8b.
    return PrefixedEmbeddings(
        HuggingFaceEmbeddings(model_name=MEMORY_EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    )
