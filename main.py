from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import RunnableConfig

from graph import ModelGraph
from tools import *

model = ChatOllama(model="qwen3:8b")

checkpointer = InMemorySaver()

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

app = ModelGraph.compile(checkpointer=checkpointer)

while True:
    query = str(input("User: "))

    if query in ["exit", "quit"]:
        break

    print("AI: ", end="", flush=True)
    for chunk, metadata in app.stream(
        {"messages": [HumanMessage(content=query)], "search_file": []},
        config=config,
        stream_mode="messages",
    ):
        if (
            metadata.get("langgraph_node")  # pyright: ignore
            in [
                "direct",
                "rag",
                "code_tools",
                "tool_execute",
            ]
            and chunk.content  # pyright: ignore
        ):
            print(chunk.content, end="", flush=True)  # pyright: ignore

    print("\n")
