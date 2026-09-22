import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.environ["MODEL"]

direct_prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant."), ("user", "{question}")]
)

model = ChatOllama(model=MODEL_NAME)

direct_chain = direct_prompt | model
