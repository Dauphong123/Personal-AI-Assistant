"""Many web conversations over one compiled graph.

Each ChatSession has its own LangGraph thread and event log. A shared semaphore limits how
many turns run on the model at once (default 1: this machine runs one 8B model at a few
tokens per second, and the shared tokenizer is not known to be safe under concurrent use);
a session whose turn must wait emits a `waiting` event.
"""

import threading
from typing import Literal

from turns import run_turn

from .session import ChatSession

DEFAULT_MAX_SESSIONS = 50


class SessionManager:
    def __init__(self, app, run=run_turn, max_turns: int = 1, max_sessions: int = DEFAULT_MAX_SESSIONS):
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")

        self.app = app
        self._run = run
        self._slot = threading.Semaphore(max_turns)
        self._max_sessions = max_sessions
        self._lock = threading.Lock()
        self._sessions: dict[str, ChatSession] = {}  # insertion order = creation order

    def create(self, title: str | None = None) -> ChatSession | None:
        """A new empty session, or None when the limit is reached."""
        with self._lock:
            if len(self._sessions) >= self._max_sessions:
                return None

            session = ChatSession(self.app, run=self._run, turn_slot=self._slot, title=title)
            self._sessions[session.id] = session
            return session

    def get(self, session_id: str) -> ChatSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def list(self) -> list[dict]:
        with self._lock:
            sessions = list(self._sessions.values())

        return [s.info() for s in sessions]

    def delete(self, session_id: str) -> Literal["deleted", "missing", "busy"]:
        with self._lock:
            session = self._sessions.get(session_id)

            if session is None:
                return "missing"

            if session.busy:
                return "busy"

            del self._sessions[session_id]
            return "deleted"

    def stop_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())

        for session in sessions:
            session.stop()
