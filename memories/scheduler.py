"""Background memory saves that never hold up the chat.

Learning from a turn costs one or two calls to the main model (~32 s on qwen3:8b here) and
Ollama runs one generation at a time, so a save that runs while the user sends the next message
makes that turn wait. A save therefore starts only when no turn is running and the chat has been
idle for `MEMORY_IDLE_SECONDS`, and a turn that starts cancels the save's model call (Ollama stops
generating when the request is closed); the save is retried from the start at the next idle
moment. Nothing is written before the model calls finish, so a cancelled save leaves no trace.
One worker keeps saves in order.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from settings import memory_idle_seconds

from .extractor import MemoryCancelled

logger = logging.getLogger(__name__)

IDLE_SECONDS = memory_idle_seconds()


class MemoryScheduler:
    def __init__(self, idle_seconds: float = IDLE_SECONDS, clock=time.monotonic):
        self.idle_seconds = idle_seconds
        self._clock = clock
        self._cond = threading.Condition()
        self._active = 0
        self._last = float("-inf")
        self._flushing = 0
        self._cancel: threading.Event | None = None  # of the save that is running
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory")

    def turn_started(self) -> None:
        with self._cond:
            self._active += 1
            self._last = self._clock()

            if self._cancel is not None:
                self._cancel.set()

    def turn_finished(self) -> None:
        with self._cond:
            self._active = max(0, self._active - 1)
            self._last = self._clock()
            self._cond.notify_all()

    def submit(self, manager, messages: list, thread_id: str | None) -> None:
        self._pool.submit(self._run, manager, messages, thread_id)

    def _start(self) -> threading.Event:
        """Wait for an idle moment, then hand out the cancel event of the save that starts."""
        with self._cond:
            while not self._flushing:
                if self._active:
                    wait = None
                else:
                    wait = self._last + self.idle_seconds - self._clock()
                    if wait <= 0:
                        break

                self._cond.wait(wait)

            self._cancel = threading.Event()
            return self._cancel

    def _run(self, manager, messages, thread_id) -> None:
        while True:
            cancel = self._start()

            try:
                manager.remember(messages, thread_id=thread_id, cancel=cancel)
                return
            except MemoryCancelled:
                logger.info("Memory save paused for a new turn; it runs again when the chat is idle")
            except Exception:
                logger.warning("Saving memory failed", exc_info=True)
                return
            finally:
                with self._cond:
                    self._cancel = None

    def wait(self, timeout: float = 60) -> bool:
        """Run the queued saves now, without waiting for idle (call before the program exits)."""
        with self._cond:
            self._flushing += 1
            self._cond.notify_all()

        try:
            self._pool.submit(lambda: None).result(timeout=timeout)  # queued behind earlier saves
            return True
        except FutureTimeout:
            logger.warning("Memory is still being saved; exiting without waiting longer")
            return False
        finally:
            with self._cond:
                self._flushing -= 1


MEMORY_SCHEDULER = MemoryScheduler()
