import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from chains import code_chain, direct_chain, router_chain, summerize_chain
from data.route_query import Router
from data.state import State
from dependency import dependency
from tools import NEEDS_APPROVAL, TOOL_BY_NAME

logger = logging.getLogger(__name__)

# The model's context window is 8192 tokens by default and the coding prompt with its tool
# schemas takes about 1,400 of them, plus room for the answer. A longer history makes Ollama
# silently drop the oldest messages (possibly splitting a tool call from its result).
MAX_HISTORY_TOKENS = int(os.environ.get("MAX_HISTORY_TOKENS", "5000"))
MAX_TOOL_CALLS_PER_TURN = 30
MAX_TOOL_OUTPUT_CHARS = 20000
ROUTER_HISTORY_MESSAGES = 4


def _message_text(message) -> str:
    """What the model reads for this message: its text plus the arguments of its tool calls
    (an assistant message that only calls tools has no text, but its arguments can be large)."""
    text = str(message.content)

    for call in getattr(message, "tool_calls", None) or []:
        text += f" {call['name']} {json.dumps(call['args'], ensure_ascii=False)}"

    return text


def count_message_tokens(messages):
    return sum(
        len(dependency["tokenizer"].encode(_message_text(message), add_special_tokens=False))
        for message in messages
    )


def _last_human_index(messages) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return index

    return None


def recall_memory(state: State):
    manager = dependency.get("memory")
    context = ""

    last = state["messages"][-1]

    if manager is not None and not last.additional_kwargs.get("notice"):
        try:
            context = manager.recall(str(last.content))
        except Exception:
            logger.warning("Memory recall failed", exc_info=True)

    return {"memory_context": context}


# Learning from a turn costs a whole extra model call, so it runs in the background: the turn
# ends (and the next message can start) without waiting for it. One worker keeps saves in order.
_memory_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory")


def _remember(manager, messages, thread_id) -> None:
    try:
        manager.remember(messages, thread_id=thread_id)
    except Exception:
        logger.warning("Saving memory failed", exc_info=True)


def save_memory(state: State, config: RunnableConfig):
    manager = dependency.get("memory")

    if manager is not None:
        thread_id = config.get("configurable", {}).get("thread_id")
        _memory_pool.submit(_remember, manager, list(state["messages"]), thread_id)

    return {}


def wait_for_memory(timeout: float = 60) -> bool:
    """Let saves that are still running finish (call before the program exits)."""
    try:
        _memory_pool.submit(lambda: None).result(timeout=timeout)  # queued behind earlier saves
        return True
    except FutureTimeout:
        logger.warning("Memory is still being saved; exiting without waiting longer")
        return False


def code_tools(state: State):
    response = code_chain.invoke(
        {
            "messages": state["messages"],
            "memory": state.get("memory_context", ""),
        }
    )

    return {"messages": [response], "search_file": []}


def has_tools(state: State):
    if state["messages"][-1].tool_calls:
        return "tool_execute"

    return "continue"


def _run_tool(call) -> str:
    tool = TOOL_BY_NAME.get(call["name"])

    if tool is None:
        return f"ERROR: unknown tool '{call['name']}'. Available: {', '.join(TOOL_BY_NAME)}"

    try:
        output = str(tool.invoke(call["args"]))
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

    if len(output) > MAX_TOOL_OUTPUT_CHARS:
        output = output[:MAX_TOOL_OUTPUT_CHARS] + "\n... output truncated"

    return output


def tool_execute(state: State):
    calls = state["messages"][-1].tool_calls

    if not calls:
        return {"messages": []}

    human = _last_human_index(state["messages"])
    used = sum(isinstance(m, ToolMessage) for m in state["messages"][(human or 0) :])

    # Ask once, before anything runs: on resume this node re-executes from the top,
    # so nothing with side effects may happen before the interrupt.
    risky = [c for c in calls if c["name"] in NEEDS_APPROVAL]
    approved = True

    if risky and used < MAX_TOOL_CALLS_PER_TURN:
        approved = bool(
            interrupt(
                {"calls": [{"name": c["name"], "args": c["args"]} for c in risky]}
            )
        )

    messages = []
    for call in calls:
        if used >= MAX_TOOL_CALLS_PER_TURN:
            output = "ERROR: tool call limit reached for this turn. Answer with what you have."
        elif call["name"] in NEEDS_APPROVAL and not approved:
            output = "ERROR: the user declined this action."
        else:
            output = _run_tool(call)

        messages.append(ToolMessage(content=output, tool_call_id=call["id"]))

    return {"messages": messages, "search_file": []}


def router(state: State):
    messages = state["messages"]
    history = "\n".join(
        f"{m.type}: {str(m.content)[:300]}"
        for m in messages[-(ROUTER_HISTORY_MESSAGES + 1) : -1]
    )

    response: Router = router_chain.invoke(  # pyright: ignore
        {"question": str(messages[-1].content), "history": history}
    )

    if response.destination == "code":
        return "code_tools"

    return "direct"


def direct(state: State):
    response = direct_chain.invoke(
        {
            "messages": state["messages"],
            "memory": state.get("memory_context", ""),
        }
    )

    return {"messages": [response], "search_file": []}


def summarize_need(state: State):
    tokens_num = count_message_tokens(state["messages"])

    if tokens_num > MAX_HISTORY_TOKENS:
        return "summarize"

    return "continue"


def _as_text(messages) -> str:
    lines = []

    for m in messages:
        text = str(m.content)

        if isinstance(m, ToolMessage):
            text = text[:300]

        lines.append(f"{m.type}: {text}")

    return "\n".join(lines)


def summarize(state: State):
    """Replace everything before the latest turn with a summary.

    The latest turn (from the last human message on) is kept whole, so a tool call
    is never separated from its result.
    """
    messages = state["messages"]
    split = _last_human_index(messages)

    if not split:
        # No earlier history to condense.
        return {}

    old, recent = messages[:split], messages[split:]
    summary = summerize_chain.invoke({"messages": _as_text(old)})

    return {
        # New messages are appended after removals, so the recent turn is re-added
        # (with fresh ids) behind the summary to keep the order.
        "messages": [RemoveMessage(id=m.id) for m in messages]
        + [AIMessage(content=f"Summary of the earlier conversation:\n{summary}")]
        + [m.model_copy(update={"id": str(uuid4())}) for m in recent],
        "search_file": [],
    }


def reverse_router(state: State):
    return {}
