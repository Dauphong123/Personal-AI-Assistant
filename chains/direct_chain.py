import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from dependency import dependency

load_dotenv()

MODEL_NAME = os.environ["MODEL"]

direct_prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant."), MessagesPlaceholder("messages")]
)

model = dependency["model"]

direct_chain = direct_prompt | model
