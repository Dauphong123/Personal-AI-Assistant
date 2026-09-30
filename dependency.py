import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from rag.rag import RAG


def dependency_init():
    load_dotenv()
    MODEL_NAME = os.environ["MODEL"]
    model = ChatOllama(model=MODEL_NAME)
    rag = RAG()

    return {"model": model}


dependency = dependency_init()
