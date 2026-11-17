import logging

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from transformers import AutoTokenizer

from llm_options import TOKENIZER_MAP, model_name, ollama_options
from rag.rag import RAG

logger = logging.getLogger(__name__)

load_dotenv()

MODEL_NAME = model_name()


def model_init():
    model = ChatOllama(model=MODEL_NAME, **ollama_options())
    return model


def rag_init():
    rag = RAG()
    rag.add_document("./documents/")
    return rag


def db_init():
    from memories.database import init_db

    init_db()


def memory_init(model):
    """Build the long-term memory manager, or return None if Postgres is unavailable."""
    try:
        from memories.embeddings import MEMORY_EMBEDDING_MODEL, memory_embeddings
        from memories.extractor import MemoryExtractor
        from memories.manager import MemoryManager
        from memories.store import MemoryStore

        db_init()

        store = MemoryStore(memory_embeddings())
        changed = store.ensure_embedding_model(MEMORY_EMBEDDING_MODEL)
        if changed:
            logger.info("Re-embedded %d memories with %s", changed, MEMORY_EMBEDDING_MODEL)

        return MemoryManager(store, MemoryExtractor(model))
    except Exception as exc:
        try:
            from memories.check import explain_failure

            hint = explain_failure(exc)
        except Exception:
            hint = "Is `docker compose up -d` running?"
        logger.warning("Long-term memory disabled.\n%s", hint, exc_info=True)
        return None


def tokenizer_init():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MAP[MODEL_NAME])
    return tokenizer


def dependency_init():
    model = model_init()
    rag = rag_init()
    tokenizer = tokenizer_init()
    memory = memory_init(model)

    return {"model": model, "rag": rag, "tokenizer": tokenizer, "memory": memory}


dependency = dependency_init()
