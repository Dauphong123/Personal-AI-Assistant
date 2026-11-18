import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from dependency import dependency
from llm_options import ollama_options

prompt = ChatPromptTemplate.from_template(
    """
    The conversation may already contain a previous summary
    Write the summary in the same language as the conversation.
    Create a compact summary that preserves
    Preserve:
    - important facts
    - decisions
    - user requirements
    - relevant technical context
    - unresolved questions
    - constrains
    - user's goals

    Conversation:
    {messages}
"""
)

SUMMARY_TEMPERATURE = 0.3


def summary_model(env, main):
    """The main model unless SUMMARY_MODEL names another one. Only one model fits this GPU, so a
    second model makes Ollama unload the main one and reload it (~8 s) on the next turn."""
    name = (env.get("SUMMARY_MODEL") or "").strip()

    if name:
        return ChatOllama(model=name, temperature=SUMMARY_TEMPERATURE, **ollama_options(env))

    return main.model_copy(update={"temperature": SUMMARY_TEMPERATURE})


model = summary_model(os.environ, dependency["model"])

parser = StrOutputParser()

summerize_chain = prompt | model | parser


def summerize(messages):
    return summerize_chain.invoke({"messages": messages})
