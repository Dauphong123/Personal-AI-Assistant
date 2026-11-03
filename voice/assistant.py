"""The voice interaction state machine.

Voice is only another way to feed `ChatLoop`: transcripts go to `chat.submit()` (the
same entry point as the keyboard) and final replies come back through `chat.on_reply`.

Everything audio-related runs on the thread that calls `run()`; the agent runs on
ChatLoop's worker thread. While the agent works the wake word is still monitored, so
"Jarvis, stop" cancels a running turn and "Jarvis, <request>" queues one. The wake
word is never monitored while the assistant is speaking.
"""

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from enum import Enum, auto
from typing import Protocol

import numpy as np

from language import detect_language, say
from voice.audio import MicDisconnected
from voice.config import VoiceConfig
from voice.text import is_filler, normalize_answer, strip_wake_word, wake_word_phrases

logger = logging.getLogger(__name__)

PREROLL_FRAMES = 3  # about 240 ms kept from before the detection fired
MIC_RETRIES = 5


class VoiceState(Enum):
    WAITING_FOR_WAKE_WORD = auto()
    WAKE_WORD_DETECTED = auto()
    LISTENING_FOR_COMMAND = auto()
    TRANSCRIBING = auto()
    PROCESSING = auto()
    SPEAKING = auto()
    RETURN_TO_STANDBY = auto()
    STOPPING = auto()


S = VoiceState

TRANSITIONS: dict[VoiceState, set[VoiceState]] = {
    S.WAITING_FOR_WAKE_WORD: {S.WAKE_WORD_DETECTED, S.PROCESSING, S.STOPPING},
    S.WAKE_WORD_DETECTED: {S.LISTENING_FOR_COMMAND, S.RETURN_TO_STANDBY, S.STOPPING},
    S.LISTENING_FOR_COMMAND: {S.TRANSCRIBING, S.RETURN_TO_STANDBY, S.STOPPING},
    S.TRANSCRIBING: {S.LISTENING_FOR_COMMAND, S.PROCESSING, S.RETURN_TO_STANDBY, S.STOPPING},
    # Wake word monitoring stays on while the agent works.
    S.PROCESSING: {S.WAKE_WORD_DETECTED, S.SPEAKING, S.RETURN_TO_STANDBY, S.STOPPING},
    # After an approval prompt the assistant listens for the answer straight away.
    S.SPEAKING: {S.LISTENING_FOR_COMMAND, S.RETURN_TO_STANDBY, S.STOPPING},
    S.RETURN_TO_STANDBY: {S.WAITING_FOR_WAKE_WORD, S.PROCESSING, S.STOPPING},
    S.STOPPING: set(),
}


class InvalidTransition(RuntimeError):
    pass


class ChatPort(Protocol):
    """The part of ChatLoop voice mode uses."""

    on_reply: Callable[[str], None] | None
    on_speech: Callable[[str], None] | None
    on_cancel: Callable[[], None] | None
    speech_limit: int

    @property
    def busy(self) -> bool: ...

    @property
    def pending_approval(self) -> dict | None: ...

    def submit(self, line: str) -> bool: ...


def describe_approval(request: dict, language: str = "en") -> str:
    """Short spoken form of a tool-approval request."""
    parts = []

    for call in request["calls"]:
        args = " ".join(str(v) for v in call["args"].values())[:80]
        parts.append(f"{call['name'].replace('_', ' ')} {args}".strip())

    return f"{say('approve_intro', language)} " + say("approve_join", language).join(parts) + f". {say('approve_outro', language)}"


class VoiceAssistant:
    def __init__(
        self,
        chat: ChatPort,
        mic,
        detector,
        recorder,
        transcriber,
        speaker,
        config: VoiceConfig,
        show: Callable[[str], None] = print,
        sleep: Callable[[float], None] = time.sleep,
        playback_grace: float = 0.3,
        stop_event: threading.Event | None = None,
        verifier=None,
    ):
        self.chat = chat
        self.mic = mic
        self.detector = detector
        self.recorder = recorder
        self.transcriber = transcriber
        self.speaker = speaker
        self.config = config
        self.verifier = verifier  # None: any voice is accepted
        self._show = show
        self._sleep = sleep
        self._grace = playback_grace

        self.state = S.WAITING_FOR_WAKE_WORD
        self.cycles = 0  # completed trips back to standby
        self._stop = stop_event or threading.Event()
        self._replies: queue.Queue[str] = queue.Queue()
        self._ring: deque[np.ndarray] = deque(maxlen=PREROLL_FRAMES)
        self._preroll: list[np.ndarray] = []
        self._audio: np.ndarray | None = None
        self._heard: list[np.ndarray] = []  # all audio of this interaction, for the voice check
        self._attempt = 0  # 0: right after the wake word, 1: after the "I'm listening" beep
        # (text, listen for an answer afterwards, epoch it was queued in)
        self._speech: tuple[str, bool, int] | None = None
        # Bumped when the user cancels, so speech that was already taken from the queue but not yet
        # started is dropped too (Speaker.say resets its own stop flag when it starts).
        self._speech_epoch = 0
        self._expecting_answer = False
        self._lang = "vi" if config.stt_languages[:1] == ("vi",) else "en"  # language of the last thing you said
        # The approval request already announced. Held by reference (not id()): a freed object's
        # id can be reused by the next request, which would then never be announced.
        self._asked: dict | None = None
        # Sentences arrive while the answer is still being generated and are spoken in bursts
        # between wake-word checks, so "Jarvis, stop" still works mid-answer.
        chat.on_reply = self._replies.put
        chat.on_speech = self._replies.put
        chat.on_cancel = self._drop_pending_speech
        chat.speech_limit = config.max_spoken_chars

        phrases = wake_word_phrases(config.wake_word)
        self._phrase = phrases[0] if phrases else config.wake_word

    # --- lifecycle ---------------------------------------------------------------

    def request_stop(self) -> None:
        """Ask the loop to shut down. Safe to call from any thread or a signal handler."""
        self._stop.set()
        self.speaker.stop()

    def run(self) -> None:
        try:
            if not self._open_mic():
                self._transition(S.STOPPING)
                return

            self._banner_standby()

            while self.state is not S.STOPPING:
                if self._stop.is_set():
                    self._transition(S.STOPPING)
                    break

                try:
                    self._handlers[self.state]()
                except MicDisconnected:
                    self._recover_mic()
                except Exception:
                    logger.exception("Voice loop error in state %s", self.state.name)
                    self._show("[ERROR] Something went wrong; returning to standby.")
                    self._recover()
                    self._sleep(0.5)
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        if self.state is not S.STOPPING:
            self.state = S.STOPPING

        self._show("[STOPPING] Shutting down voice mode.")

        # One failing close must not stop the other: an open microphone can keep the process alive.
        for release in (self.speaker.close, self.mic.close):
            try:
                release()
            except Exception:
                logger.exception("Releasing an audio device failed")

    # --- state plumbing ----------------------------------------------------------

    def _transition(self, new: VoiceState) -> None:
        if new not in TRANSITIONS[self.state]:
            raise InvalidTransition(f"{self.state.name} -> {new.name}")

        logger.debug("voice: %s -> %s", self.state.name, new.name)
        self.state = new

        message = {
            S.WAKE_WORD_DETECTED: "[ACTIVATED] Wake word detected.",
            S.LISTENING_FOR_COMMAND: "[LISTENING] Listening for your command...",
            S.TRANSCRIBING: "[TRANSCRIBING] Recognizing speech...",
            S.PROCESSING: "[PROCESSING] Agent is processing your request...",
            S.SPEAKING: "[SPEAKING] Playing response...",
        }.get(new)

        if message:
            self._show(message)

    def _banner_standby(self) -> None:
        self._show(f"[STANDBY] Waiting for {self._phrase.title()}...")

    @property
    def _handlers(self) -> dict[VoiceState, Callable[[], None]]:
        return {
            S.WAITING_FOR_WAKE_WORD: self._watch,
            S.PROCESSING: self._watch,
            S.WAKE_WORD_DETECTED: self._on_wake,
            S.LISTENING_FOR_COMMAND: self._listen,
            S.TRANSCRIBING: self._transcribe,
            S.SPEAKING: self._speak,
            S.RETURN_TO_STANDBY: self._standby,
        }

    def _take_pending_speech(self) -> str:
        """Everything queued so far, as one burst (the speaker still goes sentence by sentence)."""
        parts = []

        while True:
            try:
                parts.append(self._replies.get_nowait())
            except queue.Empty:
                return " ".join(parts)

    def _drop_pending_speech(self) -> None:
        """The user cancelled the work: forget what has not been spoken and cut off what is."""
        self._speech_epoch += 1
        self._take_pending_speech()
        self.speaker.stop()

    def _has_background_work(self) -> bool:
        return (
            self.chat.busy
            or not self._replies.empty()
            or self.chat.pending_approval is not None
        )

    def _recover(self) -> None:
        """Back to standby after an unexpected error, whatever state things are in."""
        try:
            self._to_standby()
        except Exception:
            logger.exception("Could not return to standby cleanly")
            self._expecting_answer = False

            if self.state is not S.STOPPING:
                self.state = S.RETURN_TO_STANDBY

    def _to_standby(self) -> None:
        """Abandon whatever is in progress (denying a pending spoken approval)."""
        if self._expecting_answer:
            self._expecting_answer = False

            try:
                self.chat.submit("n")
            except Exception:
                logger.exception("Could not decline the pending approval")

        if self.state not in (S.WAITING_FOR_WAKE_WORD, S.RETURN_TO_STANDBY, S.STOPPING):
            self._transition(S.RETURN_TO_STANDBY)

    # --- WAITING_FOR_WAKE_WORD / PROCESSING ---------------------------------------

    def _watch(self) -> None:
        frame = self.mic.read(timeout=0.1)

        if frame is not None:
            self._ring.append(frame)

            if self.detector.process(frame):
                self._preroll = list(self._ring)
                self._heard = []
                self._attempt = 0
                self._transition(S.WAKE_WORD_DETECTED)
                return

        if self.state is S.WAITING_FOR_WAKE_WORD:
            if self._has_background_work():
                self._transition(S.PROCESSING)

            return

        request = self.chat.pending_approval

        if request is not None and request is not self._asked:
            self._asked = request
            self._speech = (describe_approval(request, self._lang), True, self._speech_epoch)
            self._transition(S.SPEAKING)
        elif not self._replies.empty():
            self._speech = (self._take_pending_speech(), False, self._speech_epoch)
            self._transition(S.SPEAKING)
        elif not self.chat.busy and request is None:
            self._transition(S.RETURN_TO_STANDBY)

    # --- WAKE_WORD_DETECTED -------------------------------------------------------

    def _on_wake(self) -> None:
        # No spoken acknowledgment yet: the user may already be saying the command
        # ("Jarvis, explain ..."), and talking over a prompt would lose it.
        self._transition(S.LISTENING_FOR_COMMAND)

    # --- LISTENING_FOR_COMMAND ----------------------------------------------------

    def _listen(self) -> None:
        use_preroll = self._attempt == 0 and not self._expecting_answer
        audio = self.recorder.record(
            self.mic,
            preroll=self._preroll if use_preroll else None,
            listen_timeout=self.config.listening_timeout,
            max_duration=self.config.max_recording_duration,
        )
        self._preroll = []

        if audio is None:
            if not self._stop.is_set():
                self._show("[STANDBY] No speech heard.")

            self._to_standby()
            return

        self._audio = audio
        self._transition(S.TRANSCRIBING)

    # --- TRANSCRIBING -------------------------------------------------------------

    def _transcribe(self) -> None:
        audio, self._audio = self._audio, None

        if audio is None:  # nothing was recorded; there is nothing to recognize
            self._to_standby()
            return

        self._heard.append(audio)

        try:
            text = self.transcriber.transcribe(audio).strip()
        except Exception:
            logger.exception("Speech recognition failed")
            self._show("[ERROR] Speech recognition failed.")
            self._to_standby()
            return

        if not self._expecting_answer:
            text = strip_wake_word(text, self.config.wake_word)

        if not any(ch.isalnum() for ch in text) or (not self._expecting_answer and is_filler(text)):
            if self._attempt == 0 and not self._expecting_answer:
                # Only the wake word was heard: signal readiness and wait for the command.
                self._attempt = 1
                self._play_beep()
                self._transition(S.LISTENING_FOR_COMMAND)
            else:
                self._show("[STANDBY] Nothing recognized.")
                self._to_standby()

            return

        self._lang = detect_language(text)

        if self.verifier is not None:
            # Judge the voice on everything said so far in this interaction: the wake word
            # alone is too short, and a short answer like "yes" is too little by itself.
            result = self.verifier.verify(np.concatenate(self._heard))

            if not result.accepted:
                logger.info("Voice rejected: %s (similarity %.2f)", result.reason, result.score)
                self._show(f"[REJECTED] {result.reason.capitalize()} (similarity {result.score:.2f}).")
                self._to_standby()
                return

            logger.info("Voice accepted (similarity %.2f)", result.score)

        self._show(f"You said: {text}")

        if self._expecting_answer and self.chat.pending_approval is None:
            # Answered on the keyboard (or cancelled) while we were listening.
            self._expecting_answer = False
            self._to_standby()
            return

        if self._expecting_answer or self.chat.pending_approval is not None:
            text = normalize_answer(text)

        self._expecting_answer = False

        if not self.chat.submit(text):
            self.request_stop()  # the user said "exit" or "quit"
            self._transition(S.STOPPING)
            return

        self._transition(S.PROCESSING)

    # --- SPEAKING -----------------------------------------------------------------

    def _speak(self) -> None:
        text, listen_after, epoch = self._speech or ("", False, self._speech_epoch)
        self._speech = None

        if epoch != self._speech_epoch:
            # Cancelled after this was queued: say nothing, and do not wait for an answer to a
            # question that no longer exists.
            text, listen_after = "", False

        try:
            if text:
                self.speaker.say(text)
        except Exception:
            logger.exception("Speech output failed")
            self._show("[ERROR] Could not play the response; the text is shown above.")

        self._after_playback()

        if listen_after:
            self._expecting_answer = True
            self._heard = []
            self._attempt = 1
            self._transition(S.LISTENING_FOR_COMMAND)
        else:
            self._transition(S.RETURN_TO_STANDBY)

    def _play_beep(self) -> None:
        try:
            self.speaker.beep()
        except Exception:
            logger.exception("Could not play the activation sound")

        self._after_playback()

    def _after_playback(self) -> None:
        """Keep the assistant's own voice out of the wake-word detector."""
        self._sleep(self._grace)
        self.mic.flush()
        self._ring.clear()
        self.detector.reset()

    # --- RETURN_TO_STANDBY --------------------------------------------------------

    def _standby(self) -> None:
        self._ring.clear()
        self._preroll = []
        self._audio = None
        self._heard = []
        self._attempt = 0
        self._expecting_answer = False
        self.mic.flush()
        self.detector.reset()
        self.cycles += 1

        if self._has_background_work():
            self._transition(S.PROCESSING)
        else:
            self._transition(S.WAITING_FOR_WAKE_WORD)
            self._banner_standby()

    # --- microphone ---------------------------------------------------------------

    def _open_mic(self) -> bool:
        delay = 1.0

        for attempt in range(1, MIC_RETRIES + 1):
            try:
                self.mic.start()
                return True
            except Exception as e:
                logger.warning("Microphone unavailable (attempt %d): %s", attempt, e)
                self._show(f"[ERROR] Microphone unavailable ({e}); retry {attempt}/{MIC_RETRIES}...")

                if attempt < MIC_RETRIES:
                    self._sleep(delay)
                    delay = min(delay * 2, 8.0)

        self._show("[ERROR] Could not open the microphone. Voice mode stopped.")
        return False

    def _recover_mic(self) -> None:
        self._show("[ERROR] Microphone disconnected; trying to reconnect...")
        self.mic.close()

        if not self._open_mic():
            self.state = S.STOPPING
            return

        self.detector.reset()
        self._to_standby()
