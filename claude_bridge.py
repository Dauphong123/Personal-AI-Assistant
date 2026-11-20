"""Claude Code as a chat model (F036): runs the user's installed `claude` headless in stream-json mode,
so the user's own Claude login pays for it and Claude Code's own context handling, thinking and tools
apply. File edits and shell commands are asked about through our `approve` callback.

The stream-json message shapes follow the Claude Agent SDK protocol; they are assumed until checked
against a real run (tests/fixtures/claude_stream)."""

import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from turns import TurnCancelled

logger = logging.getLogger(__name__)

# Claude Code tools that change files or run commands: the user approves each call, as with ours.
APPROVAL_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit", "Bash"})
DECLINED = "The user declined this action."
POLL_SECONDS = 0.1
STDERR_LINES = 20


class ClaudeCodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class ToolStart:
    id: str
    name: str
    args: dict


@dataclass(frozen=True)
class ToolResult:
    id: str
    text: str
    is_error: bool


@dataclass(frozen=True)
class Permission:
    request_id: str
    tool: str
    args: dict


@dataclass(frozen=True)
class Done:
    text: str
    is_error: bool


Event = Text | ToolStart | ToolResult | Permission | Done
Approve = Callable[[dict], bool]


def session_uuid(thread_id: str) -> str:
    """One Claude Code session per chat thread, so the next turn can --resume it."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "langchain-agent/" + thread_id))


def claude_command(alias: str, effort: str, session_id: str, resume: bool, exe: str = "claude") -> list[str]:
    return [
        exe,
        "-p",
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--model", alias,
        "--effort", effort,
        # Only the project's settings: the user's global hooks, plugins and MCP servers stay out.
        "--setting-sources", "project,local",
        "--strict-mcp-config",
        # Permission prompts come to us on stdout; we answer on stdin.
        "--permission-prompts", "host",
        "--resume" if resume else "--session-id", session_id,
    ]


def _block_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and isinstance(b.get("text"), str))

    return ""


def parse_line(line: str) -> list[Event]:
    """Events in one stdout line; anything unknown or unreadable is skipped."""
    try:
        message = json.loads(line)
    except ValueError:
        return []

    if not isinstance(message, dict):
        return []

    kind = message.get("type")

    if kind == "stream_event":
        delta = (message.get("event") or {}).get("delta") or {}
        if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
            return [Text(delta["text"])]
        return []

    if kind == "control_request":
        request = message.get("request") or {}
        if request.get("subtype") == "can_use_tool":
            return [Permission(str(message.get("request_id", "")), str(request.get("tool_name", "")), dict(request.get("input") or {}))]
        return []

    if kind == "result":
        return [Done(str(message.get("result") or ""), bool(message.get("is_error")))]

    blocks = (message.get("message") or {}).get("content")
    if not isinstance(blocks, list):
        return []

    events: list[Event] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue

        # Assistant text is not taken here: it already arrived as stream deltas.
        if kind == "assistant" and block.get("type") == "tool_use":
            events.append(ToolStart(str(block.get("id", "")), str(block.get("name", "")), dict(block.get("input") or {})))
        elif kind == "user" and block.get("type") == "tool_result":
            events.append(ToolResult(str(block.get("tool_use_id", "")), _block_text(block.get("content")), bool(block.get("is_error"))))

    return events


def kill_tree(process) -> None:
    """Stop Claude Code and anything it started (a shell command, for example)."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True, check=False)
        else:
            process.kill()
    except OSError:
        logger.info("Could not stop Claude Code", exc_info=True)


def _send(process, message: dict) -> None:
    try:
        process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        process.stdin.flush()
    except (OSError, ValueError):
        logger.info("Claude Code input closed", exc_info=True)


def _answer(process, request: Permission, allow: bool) -> None:
    response = {"behavior": "allow", "updatedInput": request.args} if allow else {"behavior": "deny", "message": DECLINED}
    _send(process, {"type": "control_response", "response": {"subtype": "success", "request_id": request.request_id, "response": response}})


def _resolve(command: list[str]) -> list[str]:
    # On Windows `claude` is often a .cmd wrapper, which Popen finds only by its full path.
    found = shutil.which(command[0]) if command and not os.path.dirname(command[0]) else None
    return [found, *command[1:]] if found else command


def run_claude(
    command: list[str],
    prompt: str,
    cwd: str,
    approve: Approve,
    cancel: threading.Event,
    popen: Callable[..., Any] = subprocess.Popen,
    kill: Callable[[Any], None] = kill_tree,
) -> Iterator[Event]:
    """Run one turn; yield Text/ToolStart/ToolResult/Done. Permission requests are answered here."""
    try:
        process = popen(
            _resolve(command),
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise ClaudeCodeError("Claude Code is not installed: the claude command was not found.") from exc

    lines: queue.Queue[str | None] = queue.Queue()
    errors: deque[str] = deque(maxlen=STDERR_LINES)

    def read_out() -> None:
        try:
            for line in process.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    def read_err() -> None:
        for line in iter(process.stderr.readline, ""):
            errors.append(line.rstrip())

    threading.Thread(target=read_out, daemon=True).start()
    threading.Thread(target=read_err, daemon=True).start()

    _send(process, {"type": "user", "message": {"role": "user", "content": prompt}})
    finished = False

    def stop() -> None:
        kill(process)
        raise TurnCancelled

    try:
        while True:
            if cancel.is_set():
                stop()

            try:
                line = lines.get(timeout=POLL_SECONDS)
            except queue.Empty:
                continue

            if line is None:
                break

            for event in parse_line(line):
                if isinstance(event, Permission):
                    allow = event.tool not in APPROVAL_TOOLS or bool(
                        approve({"calls": [{"name": event.tool, "args": event.args}]})
                    )
                    if cancel.is_set():
                        stop()
                    _answer(process, event, allow)
                    continue

                if isinstance(event, Done):
                    finished = True
                    try:
                        process.stdin.close()  # no more input: Claude Code exits
                    except OSError:
                        pass
                    if event.is_error:
                        raise ClaudeCodeError(event.text or "Claude Code stopped with an error.")

                yield event

            if finished:
                break
    except GeneratorExit:
        kill(process)
        raise

    if finished:
        return

    code = process.wait()
    detail = "\n".join(line for line in errors if line).strip()
    raise ClaudeCodeError(detail or f"Claude Code stopped without an answer (exit code {code}).")


class ClaudeCodeChat(BaseChatModel):
    """One Claude Code turn as a LangChain chat model, so LangGraph streams its text like any model's.

    Tool activity arrives as empty chunks with additional_kwargs["claude_tool"]."""

    alias: str
    effort: str
    cwd: str
    session_id: str
    resume: bool
    approve: Approve
    cancel: threading.Event
    runner: Callable[..., Iterator[Event]] = run_claude

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "claude-code"

    def _prompt(self, messages: list[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return message.content if isinstance(message.content, str) else str(message.content)
        return ""

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        command = claude_command(self.alias, self.effort, self.session_id, self.resume)
        tools: dict[str, dict] = {}

        for event in self.runner(command, self._prompt(messages), self.cwd, self.approve, self.cancel):
            if isinstance(event, Text):
                chunk = AIMessageChunk(content=event.text)
            elif isinstance(event, ToolStart):
                tools[event.id] = {"id": event.id, "name": event.name, "args": event.args}
                chunk = AIMessageChunk(content="", additional_kwargs={"claude_tool": {**tools[event.id], "status": "running"}})
            elif isinstance(event, ToolResult):
                call = tools.get(event.id, {"id": event.id, "name": "tool", "args": {}})
                status = "error" if event.is_error else "done"
                chunk = AIMessageChunk(content="", additional_kwargs={"claude_tool": {**call, "status": status, "text": event.text}})
            else:
                continue

            generation = ChatGenerationChunk(message=chunk)

            if run_manager is not None:
                run_manager.on_llm_new_token(chunk.content if isinstance(chunk.content, str) else "", chunk=generation)

            yield generation

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = "".join(
            g.message.content for g in self._stream(messages, stop, run_manager, **kwargs) if isinstance(g.message.content, str)
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
