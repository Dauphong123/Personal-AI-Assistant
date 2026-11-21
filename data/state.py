from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class State(TypedDict):
    # add_messages appends instead of overwriting, which the memory nodes
    # (and RemoveMessage in clear_history) rely on.
    messages: Annotated[list, add_messages]
    memory_context: str
    route: str  # "code_tools", "direct" or "claude_code", set by the route node
    claude_session: str | None  # the Claude Code session this thread resumes (F036)
