import functools
import json
import logging
import os
import re
import time
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

import tool_args
from chains import code_chain, direct_chain, router_chain, summerize_chain
from data.route_query import Router
from data.state import State
from dependency import dependency
from language import normalize
from llm_options import _flag, ollama_options
from memories.manager import memory_query
from memories.scheduler import MEMORY_SCHEDULER
from settings import max_history_tokens
from tools import NEEDS_APPROVAL, TOOL_BY_NAME

logger = logging.getLogger(__name__)

# The model's context window is 8192 tokens by default and the coding prompt with its tool
# schemas takes about 1,400 of them, plus room for the answer. A longer history makes Ollama
# silently drop the oldest messages (possibly splitting a tool call from its result).
MAX_HISTORY_TOKENS = max_history_tokens()
MAX_TOOL_CALLS_PER_TURN = 30
MAX_TOOL_OUTPUT_CHARS = 20000
ROUTER_HISTORY_MESSAGES = 4

# What one model call may spend on messages: the context window minus the system prompt with its
# tool schemas and room for the answer. Within a turn, tool results can push the messages past it
# (history is only summarized after the turn); Ollama would then drop the start of the prompt,
# system prompt included (`--context-shift --keep 4` in its server log).
DEFAULT_CONTEXT_TOKENS = 8192
TOOL_PROMPT_TOKENS = 1400
ANSWER_RESERVE_TOKENS = 1500
TRIMMED_TOOL_CHARS = 300
TRIM_NOTE = "\n... [trimmed to fit the context window]"


def message_budget(env=None) -> int:
    context = ollama_options(env).get("num_ctx", DEFAULT_CONTEXT_TOKENS)
    return context - TOOL_PROMPT_TOKENS - ANSWER_RESERVE_TOKENS


def history_budget_warning(limit: int = MAX_HISTORY_TOKENS, env=None) -> str | None:
    """A warning when kept history alone would not fit one model call (not an error, so a
    smaller LLM_NUM_CTX keeps working; old tool results are trimmed per call anyway)."""
    budget = message_budget(env)

    if limit <= budget:
        return None

    return (
        f"MAX_HISTORY_TOKENS={limit} is more than the {budget} tokens a model call has for messages "
        f"(LLM_NUM_CTX minus the tool prompt and the answer); lower it or raise LLM_NUM_CTX, "
        "or Ollama may drop the oldest messages."
    )


if (_warning := history_budget_warning()) is not None:
    logger.warning(_warning)


def timing_level(env=None) -> int:
    """LOG_TIMING=true prints how long each graph step took (and Ollama's token counts)."""
    env = os.environ if env is None else env
    return logging.INFO if _flag(env, "LOG_TIMING") else logging.WARNING


timing_logger = logging.getLogger("graph.timing")
timing_logger.setLevel(timing_level())


def timed(fn):
    """Log a node's duration; functools.wraps keeps the signature LangGraph inspects."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            timing_logger.info("%s %.2fs", fn.__name__, time.perf_counter() - start)

    return wrapper


def ollama_stats(message) -> str:
    """Prompt and output token counts Ollama reports on a reply ("" when there are none)."""
    meta = getattr(message, "response_metadata", None) or {}
    parts = []

    if meta.get("prompt_eval_count") is not None:
        parts.append(
            f"prompt {meta['prompt_eval_count']} tok in {meta.get('prompt_eval_duration', 0) / 1e9:.1f}s"
        )

    if meta.get("eval_count"):
        seconds = meta.get("eval_duration", 0) / 1e9
        rate = meta["eval_count"] / seconds if seconds else 0
        parts.append(f"output {meta['eval_count']} tok at {rate:.1f} tok/s")

    return ", ".join(parts)


def _log_reply(name: str, message) -> None:
    stats = ollama_stats(message)
    if stats:
        timing_logger.info("%s: %s", name, stats)


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


MESSAGE_BUDGET = message_budget()


def _trimmed(message, chars: int):
    return message.model_copy(update={"content": str(message.content)[:chars] + TRIM_NOTE})


def fit_to_context(messages, budget: int):
    """The messages to send so they fit `budget` tokens; the state itself is not changed.

    Tool results are trimmed oldest first; the newest one is kept whole when the others make
    enough room, and cut to what fits otherwise. Call/result pairs and ids stay intact.
    """
    total = count_message_tokens(messages)

    if total <= budget:
        return messages

    fitted = list(messages)
    tool_indexes = [i for i, m in enumerate(fitted) if isinstance(m, ToolMessage)]

    for i in tool_indexes[:-1]:
        if total <= budget:
            return fitted

        before = count_message_tokens([fitted[i]])
        if len(str(fitted[i].content)) > TRIMMED_TOOL_CHARS:
            fitted[i] = _trimmed(fitted[i], TRIMMED_TOOL_CHARS)
            total += count_message_tokens([fitted[i]]) - before

    if total > budget and tool_indexes:
        last = tool_indexes[-1]
        original = messages[last]
        size = count_message_tokens([original])
        rest = total - count_message_tokens([fitted[last]])
        allowed = budget - rest
        chars = len(str(original.content))

        while chars > 0:
            chars = int(chars * max(allowed, 0) / max(size, 1) * 0.9)
            fitted[last] = _trimmed(original, chars)
            size = count_message_tokens([fitted[last]])
            if rest + size <= budget:
                break

        total = rest + count_message_tokens([fitted[last]])

    if total > budget:
        logger.warning("Messages still exceed the context budget (%d > %d tokens)", total, budget)

    return fitted


def with_memory(messages, memory: str):
    """The messages with recalled memory as a separate user message just before the latest one.

    Memory changes from turn to turn. Inside the system prompt it changed the start of every
    prompt, so Ollama re-read the whole history on each call (measured: 4.8 s for 2,200 tokens,
    8 s for 3,700). It is not a system message here either: Ollama's qwen3 template renders all
    system messages at the top (`.System`). As a user message only the latest turn is re-read
    (measured: 2.6 s instead of 4.8 s).
    """
    if not memory:
        return messages

    index = _last_human_index(messages)

    if index is None:
        return [*messages, HumanMessage(content=memory)]

    return [*messages[:index], HumanMessage(content=memory), *messages[index:]]


def _model_messages(state: State):
    memory = state.get("memory_context", "")
    budget = MESSAGE_BUDGET - (count_message_tokens([HumanMessage(content=memory)]) if memory else 0)
    return with_memory(fit_to_context(state["messages"], budget), memory)


def _last_human_index(messages) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return index

    return None


@timed
def recall_memory(state: State):
    manager = dependency.get("memory")
    context = ""

    last = state["messages"][-1]

    if manager is not None and not last.additional_kwargs.get("notice"):
        try:
            context = manager.recall(*memory_query(state["messages"]))
        except Exception:
            logger.warning("Memory recall failed", exc_info=True)

    return {"memory_context": context}


@timed
def save_memory(state: State, config: RunnableConfig):
    """Queue learning from this turn; `memories.scheduler` runs it when the chat is idle."""
    manager = dependency.get("memory")

    if manager is not None:
        thread_id = config.get("configurable", {}).get("thread_id")
        MEMORY_SCHEDULER.submit(manager, list(state["messages"]), thread_id)

    return {}


def wait_for_memory(timeout: float = 60) -> bool:
    """Let saves that are still queued or running finish (call before the program exits)."""
    return MEMORY_SCHEDULER.wait(timeout)


@timed
def code_tools(state: State):
    response = code_chain.invoke({"messages": _model_messages(state)})
    _log_reply("code_tools", response)

    return {"messages": [response]}


def has_tools(state: State):
    if state["messages"][-1].tool_calls:
        return "tool_execute"

    return "continue"


def _prepare(call) -> tuple[dict, str | None]:
    """The call's repaired arguments, or an error message instead (`tool_args.prepare`)."""
    tool = TOOL_BY_NAME.get(call["name"])

    if tool is None:
        return call["args"], f"ERROR: unknown tool '{call['name']}'. Available: {', '.join(TOOL_BY_NAME)}"

    return tool_args.prepare(tool, call["args"])


def _run_tool(call) -> str:
    args, error = _prepare(call)

    if error is not None:
        return error

    tool = TOOL_BY_NAME[call["name"]]

    try:
        output = str(tool.invoke(args))
    except Exception as e:
        # The model gets the error and usually retries; INFO keeps it off the chat console
        # but in LOG_FILE for whoever has to find out why a tool keeps failing.
        logger.info("Tool %s failed", call["name"], exc_info=True)
        return f"ERROR: {type(e).__name__}: {e}"

    if len(output) > MAX_TOOL_OUTPUT_CHARS:
        output = output[:MAX_TOOL_OUTPUT_CHARS] + "\n... output truncated"

    return output


@timed
def tool_execute(state: State):
    calls = state["messages"][-1].tool_calls

    if not calls:
        return {"messages": []}

    human = _last_human_index(state["messages"])
    used = sum(isinstance(m, ToolMessage) for m in state["messages"][(human or 0) :])

    # The user approves the arguments that will really run, and is not asked about a call
    # that would be refused anyway (repairing them has no side effects).
    prepared = [(call, *_prepare(call)) for call in calls]
    calls = [{**call, "args": args} for call, args, _ in prepared]
    refused = {call["id"]: error for call, _, error in prepared if error is not None}

    # Ask once, before anything runs: on resume this node re-executes from the top,
    # so nothing with side effects may happen before the interrupt.
    risky = [c for c in calls if c["name"] in NEEDS_APPROVAL and c["id"] not in refused]
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
        elif call["id"] in refused:
            output = refused[call["id"]]
        elif call["name"] in NEEDS_APPROVAL and not approved:
            output = "ERROR: the user declined this action."
        else:
            output = _run_tool(call)

        messages.append(ToolMessage(content=output, tool_call_id=call["id"]))

    return {"messages": messages}


# Requests that obviously need the repository or the machine, and messages that obviously do not.
# Only clear cases: anything else, short follow-ups ("yes, do it") above all, goes to the model.
_FILE_NAME = re.compile(
    r"[\w\-/\\]\w*\.(py|pyi|ts|tsx|js|jsx|json|md|txt|toml|ya?ml|ini|cfg|sh|ps1|bat|html|css|sql|csv|lock|env)\b",
    re.IGNORECASE,
)
_COMMANDS = re.compile(
    r"\b(git\s+(status|diff|log|show|commit|push|pull|branch|checkout|switch|add|stash|blame)|"
    r"pytest|unittest|pyright|npm\s+(run|install|test|ci)|pip\s+install|docker\s+compose|"
    r"run\s+(the\s+)?(tests?|build|linter|script|command)|git\s+diff)\b",
    re.IGNORECASE,
)
_FILE_ACTIONS = re.compile(
    r"\b(read|open|edit|create|delete|remove|rename|move|search|grep|list)\b"
    r"(?!.*\b(in|with|using)\s+(python|javascript|js|java|c\+\+|go|rust)\b)"
    r".*\b(file|files|folder|directory|repo|repository|codebase|project)\b",
    re.IGNORECASE,
)
_VIETNAMESE_ACTIONS = re.compile(
    r"\b(chạy\s+(lệnh|test|thử|kiểm\s+thử)|(sửa|tạo|xóa|xoá|đọc|mở|đổi\s+tên)\s+(file|tệp|thư\s+mục)|"
    r"sửa\s+(lỗi|code|mã))",
    re.IGNORECASE,
)
_SMALL_TALK = {
    "hi", "hello", "hey", "thanks", "thank you", "thanks a lot", "good morning", "good evening",
    "xin chào", "chào", "chào bạn", "cảm ơn", "cảm ơn bạn", "cám ơn",
}


def quick_route(text: str) -> str | None:
    """"code_tools" or "direct" when the message decides it alone, else None (ask the model)."""
    text = normalize(text).strip()
    bare = re.sub(r"[^\w\s]", "", text.lower()).strip()

    if bare in _SMALL_TALK:
        return "direct"

    for pattern in (_FILE_NAME, _COMMANDS, _FILE_ACTIONS, _VIETNAMESE_ACTIONS):
        if pattern.search(text):
            return "code_tools"

    return None


@timed
def router(state: State):
    messages = state["messages"]
    last = messages[-1]

    if not last.additional_kwargs.get("notice"):
        shortcut = quick_route(str(last.content))
        if shortcut:
            timing_logger.info("router: rule -> %s", shortcut)
            return shortcut
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


def route(state: State):
    """Runs beside recall_memory (the router does not need memory); `chosen_route` reads it."""
    return {"route": router(state)}


def chosen_route(state: State):
    return state.get("route") or "direct"


def routed(state: State):
    """Join point: waits for both recall_memory and route."""
    return {}


@timed
def direct(state: State):
    response = direct_chain.invoke({"messages": _model_messages(state)})
    _log_reply("direct", response)

    return {"messages": [response]}


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


@timed
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
    }

