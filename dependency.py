import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from transformers import AutoTokenizer

from rag.rag import RAG

TOKENIZER_MAP = {"qwen3:8b": "Qwen/Qwen3-8B"}


def dependency_init():
    load_dotenv()
    MODEL_NAME = os.environ["MODEL"]
    model = ChatOllama(model=MODEL_NAME)
    rag = RAG()
    rag.add_document("./documents/")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MAP[MODEL_NAME])

    return {"model": model, "rag": rag, "tokenizer": tokenizer}


dependency = dependency_init()
