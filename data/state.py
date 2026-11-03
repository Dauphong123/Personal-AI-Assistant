from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class State(TypedDict):
    # add_messages appends instead of overwriting, which the memory nodes
    # (and RemoveMessage in clear_history) rely on.
    messages: Annotated[list, add_messages]
    search_file: list
    memory_context: str
