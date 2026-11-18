from langgraph.graph import END, START, StateGraph

from data.state import State

from .node import (
    chosen_route,
    code_tools,
    direct,
    has_tools,
    recall_memory,
    reverse_router,
    route,
    routed,
    save_memory,
    summarize,
    summarize_need,
    tool_execute,
)

graph = StateGraph(State)

graph.add_node("recall_memory", recall_memory)
graph.add_node("route", route)
graph.add_node("routed", routed)
graph.add_node("save_memory", save_memory)
graph.add_node("direct", direct)
graph.add_node("code_tools", code_tools)
graph.add_node("tool_execute", tool_execute)
graph.add_node("reverse_router", reverse_router)
graph.add_node("summarize", summarize)

# Recall and routing are independent, so they run in parallel; `routed` waits for both.
graph.add_edge(START, "recall_memory")
graph.add_edge(START, "route")
graph.add_edge(["recall_memory", "route"], "routed")

graph.add_conditional_edges(
    "routed", chosen_route, {"code_tools": "code_tools", "direct": "direct"}
)

graph.add_conditional_edges(
    "code_tools",
    has_tools,
    {"tool_execute": "tool_execute", "continue": "save_memory"},
)

graph.add_edge("tool_execute", "code_tools")

graph.add_edge("direct", "save_memory")

graph.add_edge("save_memory", "reverse_router")

graph.add_conditional_edges(
    "reverse_router", summarize_need, {"summarize": "summarize", "continue": END}
)

graph.add_edge("summarize", END)
