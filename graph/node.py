from langchain_core.messages import RemoveMessage, ToolMessage

from chains import code_chain, direct_chain, rag_chain, router_chain, summerize_chain
from data.rag_data import RagOutput
from data.route_query import Router
from data.state import State
from dependency import dependency
from tools import (
    git_diff,
    list_files,
    read_file,
    search_code,
    terminal,
    delete_file,
    git_log,
    git_show,
    git_status,
    edit_file,
    move_file,
    write_file,
)

MAX_HISTORY_TOKENS = 8000


def code_tools(state: State):
    response = code_chain.invoke({"messages": state["messages"]})

    return {"messages": [response], "search_file": []}


def rag(state: State):
    output: RagOutput = rag_chain.invoke({"question": state["messages"][-1]})  # pyright: ignore

    return {"messages": [output.response], "search_file": [output.source]}


def has_tools(state: State):
    if state["messages"][-1].tool_calls:
        return "tool_execute"

    return "continue"


def tool_execute(state: State):
    tool_by_name = {
        # show project structure
        "list_files": list_files,
        "read_file": read_file,
        "search_code": search_code,
        # edit project structure
        "delete_file": delete_file,
        "edit_file": edit_file,
        "write_file": write_file,
        # using terminal
        "terminal": terminal,
        # look at git history
        "git_diff": git_diff,
        "git_log": git_log,
        "git_status": git_status,
        "git_show": git_show,
    }

    messages = []

    if not state["messages"][-1].tool_calls:
        return {"messages": []}

    for call in state["messages"][-1].tool_calls:
        tool = tool_by_name[call["name"]]
        response = tool.invoke(call["args"])
        messages.append(ToolMessage(content=str(response), tool_call_id=call["id"]))
    return {"messages": messages, "search_file": []}


def router(state: State):
    response: Router = router_chain.invoke({"question": state["messages"][-1]})  # pyright: ignore

    if response.destination == "rag":
        return "rag"

    if response.destination == "code":
        return "code_tools"

    return "direct"


def direct(state: State):
    response = direct_chain.invoke({"messages": state["messages"]})

    return {"messages": [response], "search_file": []}


def summarize_need(state: State):
    tokens_num = dependency["model"].get_num_tokens_from_messages(state["messages"])

    if tokens_num > MAX_HISTORY_TOKENS:
        return "summarize"

    return "continue"


def summarize(state: State):
    last_message = state["messages"][-1]
    response = summerize_chain.invoke({"messages": state["messages"][:-1]})

    # add both the summary and last messages to the stack
    return {"messages": [response] + [last_message], "search_file": []}


def clear_history(state: State):
    return {
        # remove all the messages except for 2 previous messages
        "messages": [RemoveMessage(id=m.id) for m in state["messages"][:-2]],
        "search_file": [],
    }


def reverse_router(state: State):
    return {}
