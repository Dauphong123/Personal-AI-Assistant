from langgraph.graph import END, START, StateGraph

from data.state import State

from .node import (
    clear_history,
    code_tools,
    direct,
    has_tools,
    rag,
    reverse_router,
    router,
    summarize,
    summarize_need,
    tool_execute,
)

graph = StateGraph(State)

graph.add_node("direct", direct)
graph.add_node("code_tools", code_tools)
graph.add_node("rag", rag)
graph.add_node("tool_execute", tool_execute)
graph.add_node("summarize_need", summarize_need)
graph.add_node("reverse_router", reverse_router)
graph.add_node("clear_history", clear_history)
graph.add_node("summarize", summarize)

graph.add_conditional_edges(
    START, router, {"rag": "rag", "code_tools": "code_tools", "direct": "direct"}
)

graph.add_conditional_edges(
    "code_tools",
    has_tools,
    {"tool_execute": "tool_execute", "continue": "reverse_router"},
)

graph.add_edge("tool_execute", "code_tools")

graph.add_edge("direct", "reverse_router")

graph.add_edge("rag", "reverse_router")

graph.add_conditional_edges(
    "reverse_router", summarize_need, {"summarize": "clear_history", "continue": END}
)

graph.add_edge("clear_history", "summarize")

graph.add_edge("summarize", END)
