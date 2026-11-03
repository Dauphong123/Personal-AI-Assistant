import logging
import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from transformers import AutoTokenizer

from llm_options import ollama_options
from rag.rag import RAG

logger = logging.getLogger(__name__)

load_dotenv()

MODEL_NAME = os.environ["MODEL"]
TOKENIZER_MAP = {"qwen3:8b": "Qwen/Qwen3-8B"}


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


def memory_init(model, rag):
    """Build the long-term memory manager, or return None if Postgres is unavailable."""
    try:
        from memories.extractor import MemoryExtractor
        from memories.manager import MemoryManager
        from memories.store import MemoryStore

        db_init()

        return MemoryManager(MemoryStore(rag.embeddings), MemoryExtractor(model))
    except Exception:
        logger.warning(
            "Long-term memory disabled (is `docker compose up -d` running?)",
            exc_info=True,
        )
        return None


def tokenizer_init():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MAP[MODEL_NAME])
    return tokenizer


def dependency_init():
    model = model_init()
    rag = rag_init()
    tokenizer = tokenizer_init()
    memory = memory_init(model, rag)

    return {"model": model, "rag": rag, "tokenizer": tokenizer, "memory": memory}


dependency = dependency_init()
