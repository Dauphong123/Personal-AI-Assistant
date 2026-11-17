"""One web conversation: runs turns on a worker thread and records what happens as events.

Events are dicts with a growing `id` and a `type`: user, waiting, token, approval, done,
cancelled, error, reset. A client reads them with `events_after(last_id)`, so a reconnect
resumes.
"""

import logging
import queue
import re
import threading
import time
import uuid

from langgraph.errors import GraphRecursionError

from turns import TurnCancelled, run_turn

logger = logging.getLogger(__name__)

DEFAULT_TITLE = "New chat"
MAX_TITLE = 60
POLL_SECONDS = 0.2


def short_title(text: str) -> str:
    """One line of at most MAX_TITLE characters (an ellipsis marks a cut)."""
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= MAX_TITLE else text[: MAX_TITLE - 1] + "…"


class ChatSession:
    def __init__(self, app, run=run_turn, turn_slot: threading.Semaphore | None = None, title: str | None = None):
        self.app = app
        self._run = run
        self._slot = turn_slot  # shared between sessions: how many turns may run at once
        self.id = uuid.uuid4().hex[:12]
        self.created = time.time()
        self._title = short_title(title) if title and title.strip() else ""
        self._cond = threading.Condition()
        self._events: list[dict] = []
        self._next_id = 1
        self._busy = False
        self._waiting = False
        self._cancel = threading.Event()
        self._approval: queue.Queue[bool] | None = None
        self._approval_request: dict | None = None
        self.thread_id = self._new_thread_id()

    def _new_thread_id(self) -> str:
        return f"web-{self.id}-{uuid.uuid4().hex[:6]}"

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def title(self) -> str:
        return self._title or DEFAULT_TITLE

    @property
    def pending_approval(self) -> dict | None:
        return self._approval_request if self._approval is not None else None

    def rename(self, title: str) -> bool:
        """Name the session. False for a blank name."""
        if not title.strip():
            return False

        self._title = short_title(title)
        return True

    def info(self) -> dict:
        if self._approval is not None:
            state = "approval"
        elif self._waiting:
            state = "waiting"
        elif self._busy:
            state = "running"
        else:
            state = "idle"

        with self._cond:
            last = self._events[-1]["id"] if self._events else 0

        return {"id": self.id, "title": self.title, "state": state, "created": self.created, "last_event": last}

    # --- events ------------------------------------------------------------------

    def _emit(self, type_: str, **fields) -> None:
        with self._cond:
            self._events.append({"id": self._next_id, "type": type_, **fields})
            self._next_id += 1
            self._cond.notify_all()

    def events_after(self, after: int, timeout: float = 0) -> list[dict]:
        """Events newer than `after`; waits up to `timeout` seconds if there are none yet."""
        with self._cond:
            if not any(e["id"] > after for e in self._events) and timeout > 0:
                self._cond.wait(timeout)
            return [e for e in self._events if e["id"] > after]

    # --- commands (called from HTTP handler threads) -------------------------------

    def send(self, text: str) -> bool:
        """Start a turn. False if the text is blank or a turn is already running."""
        text = text.strip()

        if not text:
            return False

        with self._cond:
            if self._busy:
                return False

            self._busy = True

        if not self._title:
            self._title = short_title(text)

        self._cancel.clear()
        self._emit("user", text=text)
        threading.Thread(target=self._work, args=(text,), daemon=True).start()
        return True

    def approve(self, allow: bool) -> bool:
        """Answer the waiting tool-approval request. False if none is waiting."""
        box = self._approval

        if box is None:
            return False

        box.put(bool(allow))
        return True

    def stop(self) -> None:
        """Cancel the running or waiting turn (no effect when idle)."""
        if self._busy:
            self._cancel.set()

    def new_chat(self) -> bool:
        """Forget the conversation (the title stays). Refused while a turn is running."""
        with self._cond:
            if self._busy:
                return False

            self._events.clear()
            self.thread_id = self._new_thread_id()

        self._emit("reset")
        return True

    # --- worker ------------------------------------------------------------------

    def _ask_approval(self, request: dict) -> bool:
        box: queue.Queue[bool] = queue.Queue()
        self._approval_request = request
        self._approval = box
        self._emit("approval", calls=request.get("calls", []))

        try:
            while True:
                try:
                    return box.get(timeout=POLL_SECONDS)
                except queue.Empty:
                    if self._cancel.is_set():
                        raise TurnCancelled from None
        finally:
            self._approval = None
            self._approval_request = None

    def _take_slot(self) -> bool:
        """Wait for a free turn slot; False if the turn was stopped while waiting."""
        if self._slot is None or self._slot.acquire(blocking=False):
            return True

        self._waiting = True
        self._emit("waiting")

        try:
            while not self._slot.acquire(timeout=POLL_SECONDS):
                if self._cancel.is_set():
                    return False

            return True
        finally:
            self._waiting = False

    def _work(self, text: str) -> None:
        if not self._take_slot():
            self._finish(("cancelled", {}))
            return

        try:
            outcome = self._turn(text)
        finally:
            if self._slot is not None:
                self._slot.release()

        self._finish(outcome)

    def _turn(self, text: str) -> tuple[str, dict]:
        """Run the turn; return the final event (type, fields)."""
        config = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 100}

        try:
            self._run(
                self.app,
                text,
                config,
                lambda chunk: self._emit("token", text=chunk),
                self._ask_approval,
                self._cancel,
            )
            return "done", {}
        except TurnCancelled:
            return "cancelled", {}
        except GraphRecursionError:
            return "error", {"text": "Stopped: too many steps in one turn. Try a narrower request."}
        except Exception as exc:
            logger.exception("Web turn failed")
            return "error", {"text": f"{type(exc).__name__}: {exc}"}

    def _finish(self, outcome: tuple[str, dict]) -> None:
        # The slot is already free; free the session before announcing the end, so a client
        # that reacts to the event can send its next message at once.
        type_, fields = outcome
        self._busy = False
        self._emit(type_, **fields)
