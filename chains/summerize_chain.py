import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

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

model = ChatOllama(
    model=os.environ.get("SUMMARY_MODEL", "qwen3:4b"), temperature=0.3, **ollama_options()
)

parser = StrOutputParser()

summerize_chain = prompt | model | parser


def summerize(messages):
    return summerize_chain.invoke({"messages": messages})
