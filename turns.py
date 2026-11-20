"""One conversation turn against the compiled graph, independent of any UI."""

import re
import threading
from collections.abc import Callable

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.types import Command

from language import detect_language, normalize, say
from memories.scheduler import MEMORY_SCHEDULER

STREAMED_NODES = {"direct", "code_tools", "tool_execute", "claude_code"}
# Only these produce text meant for the user; tool_execute streams raw tool output.
SPOKEN_NODES = {"direct", "code_tools", "claude_code"}

CANCELLED_TOOL_OUTPUT = "ERROR: cancelled by the user."

# Only short, unambiguous messages count: "stop the docker container" is a request.
# Vietnamese: dừng/ngừng (stop), hủy (cancel), thôi (enough), plus the accent-less spellings
# people type on a plain keyboard. Bare "dung" is not accepted: it also means "use" and "correct".
_STOP_RE = re.compile(
    r"^(?:(?:ok|okay|please|hey|wait|no|just|làm\s+ơn|vui\s+lòng|khoan|này|ơi)\s+)*"
    r"(?:stop|cancel|abort|halt|never\s*mind|forget\s+it"
    r"|dừng|ngừng|hủy|huỷ|thôi|bỏ\s+đi|đừng\s+làm\s+nữa"
    r"|dung\s+lai|ngung\s+lai|huy\s+bo|thoi\s+di)"
    r"(?:\s+(?:it|that|this|now|please|everything|all|the\s+task|current\s+task"
    r"|lại|bỏ|đi|ngay|ngay\s+lập\s+tức|nhé|nha|nữa|bây\s+giờ|luôn|giùm|làm\s+ơn|vui\s+lòng"
    r"|việc\s+này|việc\s+đó|tất\s+cả|nhiệm\s+vụ|lệnh\s+này|yêu\s+cầu))*$"
)


class TurnCancelled(Exception):
    pass


def is_stop_request(text: str) -> bool:
    cleaned = re.sub(r"[^\w\s/]", " ", normalize(text).lower())
    cleaned = " ".join(cleaned.split())
    return cleaned == "/stop" or bool(_STOP_RE.match(cleaned))


def repair_dangling_tool_calls(app, config) -> None:
    """Answer tool calls left unanswered by a cancelled turn.

    Models reject a history where an assistant tool call has no tool result.
    """
    messages = app.get_state(config).values.get("messages", [])
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    missing = [
        call["id"]
        for m in messages
        if isinstance(m, AIMessage)
        for call in m.tool_calls
        if call["id"] not in answered
    ]

    if missing:
        app.update_state(
            config,
            {
                "messages": [
                    ToolMessage(content=CANCELLED_TOOL_OUTPUT, tool_call_id=id)
                    for id in missing
                ]
            },
            as_node="tool_execute",
        )


def final_reply(app, config) -> str:
    """The user-facing answer of the finished turn: the last message, if it is plain text."""
    messages = app.get_state(config).values.get("messages", [])

    if messages and isinstance(messages[-1], AIMessage) and not messages[-1].tool_calls:
        return str(messages[-1].content)

    return ""


# A sentence ends at . ! ? or an ellipsis followed by whitespace, or at a line break.
# "3.14" and "file.py" have no whitespace after the dot, so they stay whole.
_SENTENCE_END = re.compile(r"(?<=[.!?\u2026])\s+|\n+")
_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)


def _split_complete(text: str) -> tuple[list[str], str]:
    """Complete sentences in `text`, and the unfinished tail."""
    parts = _SENTENCE_END.split(text)
    return [p for p in parts[:-1] if p.strip()], parts[-1]


class SpeechStream:
    """Turns the answer into sentences for text-to-speech while it is still being generated.

    - A message that calls tools is a working note, not the answer: it is not spoken. The
      tool call is only known after the text before it, so with `hold=True` (the coding agent)
      the newest sentence waits until the next one starts, or the message ends.
    - Code blocks are replaced by "(code omitted)".
    - Speech stops after `limit` characters ("The rest is on screen.").

    `emit` receives each sentence as soon as it is complete.
    """

    def __init__(self, emit: Callable[[str], None], limit: int = 1200):
        self._emit = emit
        self.limit = limit
        self.emitted = False
        self._buffer = ""
        self._held: str | None = None  # a complete sentence waiting to see if a tool call follows
        self._message_id = None
        self._muted = False
        self._done = False
        self._spoken = 0

    def feed(self, chunk, hold: bool = False) -> None:
        """Take one streamed message chunk (an AIMessageChunk, or a whole AIMessage)."""
        message_id = getattr(chunk, "id", None)

        if message_id != self._message_id:
            self.end_message()
            self._message_id = message_id

        if getattr(chunk, "tool_call_chunks", None) or getattr(chunk, "tool_calls", None):
            self._muted = True
            self._buffer = ""  # what was gathered so far was narration before a tool call
            self._held = None

        text = chunk.content if isinstance(chunk.content, str) else ""

        if self._muted or self._done or not text:
            return

        self._buffer += text
        self._drain(final=False, hold=hold)

    def end_message(self) -> None:
        """The current message is complete: speak what is left of it, and reset."""
        if not self._muted and not self._done:
            self._drain(final=True)

        self._buffer = ""
        self._held = None
        self._muted = False
        self._message_id = None

    def _drain(self, final: bool, hold: bool = False) -> None:
        text = _CODE_BLOCK.sub(" (code omitted) ", self._buffer)
        fence = text.find("```")
        held = ""

        if fence != -1:  # a code block that has not closed yet: hold it back
            text, held = text[:fence], text[fence:]

        sentences, rest = _split_complete(text)

        if final:
            if rest.strip():
                sentences.append(rest)

            rest = held = ""  # an unterminated code block is dropped

        self._buffer = rest + held

        if self._held is not None:
            sentences.insert(0, self._held)
            self._held = None

        if hold and not final and sentences:
            self._held = sentences.pop()

        for sentence in sentences:
            self._say(sentence)

    def _say(self, sentence: str) -> None:
        sentence = sentence.strip()

        if self._done or not re.search(r"\w", sentence):
            return

        self._spoken += len(sentence)

        if self._spoken > self.limit:
            self._done = True
            sentence = say("truncated", detect_language(sentence))

        self.emitted = True
        self._emit(sentence)


class ToolActivity:
    """Reports each tool call to `report` as it starts ("running") and when its result arrives
    ("done" / "error", with the output), so a UI can show tool work apart from the answer.

    The calls are read from the coding agent's streamed message; its chunks are added up so a
    call split across chunks is reported once, with its whole arguments at the end. Kept for the
    whole turn: after an approval pause the result streams in a later `stream_turn`."""

    def __init__(self, report: Callable[[dict], None]):
        self.report = report
        self.messages: dict[str, AIMessageChunk] = {}
        self.calls: dict[str, dict] = {}

    def feed(self, node, piece) -> bool:
        """Note one streamed chunk; True if it was tool output (not text for the answer)."""
        if node == "tool_execute":
            if isinstance(piece, ToolMessage):
                self._finished(piece)
            return True

        if node == "claude_code" and isinstance(piece, AIMessageChunk):
            # Claude Code runs its own tools; the bridge reports them on empty chunks.
            tool = piece.additional_kwargs.get("claude_tool")
            if tool:
                self.report(dict(tool))
                return True
            return False

        if node == "code_tools" and isinstance(piece, AIMessageChunk):
            key = piece.id or ""
            total = self.messages[key] + piece if key in self.messages else piece
            self.messages[key] = total

            for call in total.tool_calls:
                call_id = call.get("id")
                if call_id and call.get("name"):
                    known = call_id in self.calls
                    self.calls[call_id] = {"name": call["name"], "args": call.get("args") or {}}
                    if not known:
                        self.report({"id": call_id, **self.calls[call_id], "status": "running"})

        return False

    def _finished(self, message: ToolMessage) -> None:
        text = message.content if isinstance(message.content, str) else str(message.content)
        call = self.calls.get(message.tool_call_id, {"name": message.name or "tool", "args": {}})
        status = "error" if text.startswith("ERROR:") else "done"
        self.report({"id": message.tool_call_id, **call, "status": status, "text": text})


def stream_turn(
    app,
    graph_input,
    config,
    on_text: Callable[[str], None],
    cancel: threading.Event,
    speech: SpeechStream | None = None,
    activity: ToolActivity | None = None,
) -> None:
    stream = app.stream(graph_input, config=config, stream_mode="messages")
    last_node = None

    try:
        for chunk, metadata in stream:
            if cancel.is_set():
                raise TurnCancelled

            node = metadata.get("langgraph_node")  # pyright: ignore

            if speech is not None:
                if node != last_node:
                    speech.end_message()
                    last_node = node

                if node in SPOKEN_NODES:
                    speech.feed(chunk, hold=node in ("code_tools", "claude_code"))

            if activity is not None and activity.feed(node, chunk):
                continue

            if node in STREAMED_NODES and chunk.content:  # pyright: ignore
                on_text(chunk.content)  # pyright: ignore
    finally:
        stream.close()

    if cancel.is_set():
        raise TurnCancelled

    if speech is not None:
        speech.end_message()


def run_turn(
    app,
    query: str,
    config,
    on_text: Callable[[str], None],
    approve: Callable[[dict], bool],
    cancel: threading.Event | None = None,
    on_speech: Callable[[str], None] | None = None,
    speech_limit: int = 1200,
    notice: bool = False,
    on_tool: Callable[[dict], None] | None = None,
    model=None,
) -> bool:
    """Run one user message to completion.

    `model` (a model_choice.ModelChoice) picks the model for this turn; None means MODEL. It goes
    into the run config with `approve` and `cancel`, which a Claude Code turn uses mid-node.

    `approve` is asked about tool calls that need approval. Raises TurnCancelled
    if `cancel` is set; the state stays consistent for the next turn.

    `on_speech` receives the answer sentence by sentence while it streams (for
    text-to-speech). Returns True if any sentence was handed to it.

    `notice` marks the message as a system notice (a finished background job), not something
    the user said; memory skips such turns.

    `on_tool` receives each tool call as it starts and finishes (see `ToolActivity`); with it,
    tool output no longer goes to `on_text`.
    """
    # A turn pauses background memory saves: they share the one Ollama model with it.
    MEMORY_SCHEDULER.turn_started()
    try:
        cancel = cancel or threading.Event()
        configurable = {**config.get("configurable", {}), "model": model, "approve": approve, "cancel": cancel}
        config = {**config, "configurable": configurable}
        return _run_turn(app, query, config, on_text, approve, cancel, on_speech, speech_limit, notice, on_tool)
    finally:
        MEMORY_SCHEDULER.turn_finished()


def _run_turn(app, query, config, on_text, approve, cancel, on_speech, speech_limit, notice, on_tool=None) -> bool:
    cancel = cancel or threading.Event()
    speech = SpeechStream(on_speech, speech_limit) if on_speech else None
    activity = ToolActivity(on_tool) if on_tool else None
    repair_dangling_tool_calls(app, config)
    message = HumanMessage(content=query, additional_kwargs={"notice": True} if notice else {})
    graph_input = {"messages": [message]}

    while True:
        stream_turn(app, graph_input, config, on_text, cancel, speech, activity)

        # A paused graph is waiting for the user to approve a tool call.
        pending = [i for task in app.get_state(config).tasks for i in task.interrupts]

        if not pending:
            return bool(speech and speech.emitted)

        graph_input = Command(resume=approve(pending[0].value))
