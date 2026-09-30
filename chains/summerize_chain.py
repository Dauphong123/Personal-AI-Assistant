from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

prompt = ChatPromptTemplate.from_template(
    """
    The conversation may already contain a previous summary
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

model = ChatOllama(model="qwen3:4b", temperature=1)

parser = StrOutputParser()

summerize_chain = prompt | model | parser


def summerize(messages):
    return summerize_chain.invoke({"messages": messages})
