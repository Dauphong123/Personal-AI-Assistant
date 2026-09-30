import os

from dotenv import load_dotenv
from dependency import dependency
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

load_dotenv()

MODEL_NAME = os.environ["MODEL"]

direct_prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant."), MessagesPlaceholder("messages")]
)

model = dependency["model"]

direct_chain = direct_prompt | model
