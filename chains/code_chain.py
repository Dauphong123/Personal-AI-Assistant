import os

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from tools import *

load_dotenv()

MODEL_NAME = os.environ["MODEL"]

code_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a coding assistant working with a user's repository.

            Your job is to answer questions about the repository using the available tools.

            Rules:
            - Do not guess about repository code you have not inspected.
            - Use list_files() to understand the repository structure when necessary.
            - Use search_code() to find relevant code when you don't know where something is.
            - Use read_file() to inspect the relevant files before answering.
            - Use git_diff() when the question is about recent changes.
            - You may call multiple tools if necessary.
            - After gathering enough information, answer the user's question concisely.
            """,
        ),
        ("user", "{question}"),
    ]
)

model = ChatOllama(model=MODEL_NAME)

code_agent = model.bind_tools([list_files, read_file, search_code, git_diff, terminal])

code_chain = code_prompt | code_agent

tool_by_name = {
    "list_files": list_files,
    "read_file": read_file,
    "search_code": search_code,
    "git_diff": git_diff,
    "terminal": terminal,
}


def code_loop(x):
    messages: list[BaseMessage] = code_prompt.format_messages(question=x["question"])

    while True:
        chunks = []

        for chunk in code_agent.stream(messages):
            chunks.append(chunk)

            if chunk.content:
                print(chunk.content, end="", flush=True)

        print()

        if not chunks:
            raise RuntimeError("Model returned no chunks")

        response = chunks[0]

        for chunk in chunks[1:]:
            response += chunk

        if not response.tool_calls:
            return response

        messages.append(response)

        for call in response.tool_calls:
            tool = tool_by_name[call["name"]]

            result = tool.invoke(call["args"])

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"],
                )
            )
