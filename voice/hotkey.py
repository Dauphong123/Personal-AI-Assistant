"""Push-to-talk key (Windows): hold to talk, double tap for hands-free, tap to end hands-free.

The key is read with GetAsyncKeyState (stdlib ctypes, no extra package). It is a global
read, so the key works whichever window has focus, as in Wispr Flow. A key that types
nothing on its own (Right Ctrl by default) keeps the console input line untouched.
"""

import logging
import queue
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Windows virtual-key codes of keys that are safe to hold (no text, no toggled state).
KEYS = {
    "right_ctrl": 0xA3,
    "left_ctrl": 0xA2,
    "right_alt": 0xA5,
    "right_shift": 0xA1,
    "scroll_lock": 0x91,
    "pause": 0x13,
    "insert": 0x2D,
    **{f"f{n}": 0x6F + n for n in range(1, 13)},  # F1 = 0x70
}

TAP_MAX = 0.3  # a shorter press is a tap, a longer one a hold
DOUBLE_TAP_GAP = 0.4  # the second tap must start this soon after the first ended


class PushToTalkKey:
    """Turns key up/down samples into events: "down", "hold" (released after a hold), "tap",
    "double_tap" (instead of the second tap)."""

    def __init__(self, tap_max: float = TAP_MAX, double_tap_gap: float = DOUBLE_TAP_GAP):
        self.tap_max = tap_max
        self.double_tap_gap = double_tap_gap
        self.down = False
        self._pressed_at = 0.0
        self._last_tap_end: float | None = None
        self._second_press = False

    def update(self, down: bool, now: float) -> str | None:
        if down == self.down:
            return None

        self.down = down

        if down:
            self._pressed_at = now
            self._second_press = self._last_tap_end is not None and now - self._last_tap_end <= self.double_tap_gap
            return "down"

        if now - self._pressed_at > self.tap_max:
            self._last_tap_end = None
            return "hold"

        if self._second_press:
            self._last_tap_end = None
            return "double_tap"

        self._last_tap_end = now
        return "tap"


def windows_key_reader(name: str) -> Callable[[], bool]:
    """A function telling whether the named key is down right now (Windows only)."""
    import ctypes

    code = KEYS[name]
    user32 = ctypes.WinDLL("user32")  # raises on other systems
    return lambda: bool(user32.GetAsyncKeyState(code) & 0x8000)


class KeyWatcher:
    """Polls the key on a daemon thread. Events are queued for the voice loop (`poll()`), and
    `on_event` is called at once on the watcher thread (used to cut off speech immediately)."""

    def __init__(
        self,
        is_down: Callable[[], bool],
        on_event: Callable[[str], None] | None = None,
        interval: float = 0.02,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.on_event = on_event
        self._is_down = is_down
        self._interval = interval
        self._clock = clock
        self._key = PushToTalkKey()
        self._events: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def held(self) -> bool:
        return self._key.down

    @property
    def pending(self) -> bool:
        """Events are waiting (a tap can be over before anyone looks at `held`)."""
        return not self._events.empty()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="push-to-talk", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                event = self._key.update(self._is_down(), self._clock())
            except Exception:
                logger.warning("Reading the push-to-talk key failed; push-to-talk is off", exc_info=True)
                return

            if event:
                self._events.put(event)

                if self.on_event is not None:
                    try:
                        self.on_event(event)
                    except Exception:
                        logger.exception("Push-to-talk callback failed")

            self._stop.wait(self._interval)

    def poll(self) -> str | None:
        """The next key event, or None."""
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

    def clear(self) -> None:
        while self.poll() is not None:
            pass

    def close(self) -> None:
        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=1)
