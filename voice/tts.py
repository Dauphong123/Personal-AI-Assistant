"""Text-to-speech: Piper synthesis, and a Speaker that plays a reply sentence by sentence."""

import logging
import queue
import threading
from collections.abc import Callable
from typing import Protocol

import numpy as np

from language import detect_language
from voice.audio import make_beep
from voice.config import VoiceConfig
from voice.text import clean_for_speech, split_sentences

logger = logging.getLogger(__name__)


class SpeechError(RuntimeError):
    pass


class Synthesizer(Protocol):
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return (int16 mono samples, sample rate)."""
        ...


class Player(Protocol):
    def play(
        self,
        samples: np.ndarray,
        sample_rate: int,
        should_stop: Callable[[], bool] = ...,
    ) -> None:
        """Play a clip; return early if `should_stop()` becomes true."""
        ...

    def close(self) -> None: ...


class PiperSynthesizer:
    def __init__(self, config: VoiceConfig, voice: str | None = None):
        from piper import PiperVoice

        voice = voice or config.tts_voice
        directory = config.model_dir / "piper"
        directory.mkdir(parents=True, exist_ok=True)
        model = directory / f"{voice}.onnx"

        if not model.is_file():
            from piper.download_voices import download_voice

            logger.info("Downloading Piper voice %s", voice)
            download_voice(voice, directory)

        self._voice = PiperVoice.load(model)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        chunks = list(self._voice.synthesize(text))

        if not chunks:
            return np.zeros(0, dtype=np.int16), 22050

        samples = np.concatenate([c.audio_int16_array for c in chunks])
        return samples, chunks[0].sample_rate


class LazySynthesizer:
    """Loads a voice the first time it is used (a voice you never hear is never downloaded)."""

    def __init__(self, factory: Callable[[], Synthesizer]):
        self._factory = factory
        self._synthesizer: Synthesizer | None = None

    def load(self) -> Synthesizer:
        if self._synthesizer is None:
            self._synthesizer = self._factory()

        return self._synthesizer

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        return self.load().synthesize(text)


class Speaker:
    """Speaks replies. One reply at a time; the next sentence is synthesized while the
    current one plays, so speech starts after the first sentence rather than the last.

    `synthesizers` maps a language ("vi") to the voice for replies written in it;
    anything else uses `synthesizer`."""

    def __init__(
        self,
        synthesizer: Synthesizer,
        player: Player,
        max_chars: int = 1200,
        synthesizers: dict[str, Synthesizer] | None = None,
    ):
        self.synthesizer = synthesizer
        self.synthesizers = synthesizers or {}
        self.player = player
        self.max_chars = max_chars
        self._stop = threading.Event()
        self._lock = threading.Lock()  # prevents overlapping playback

    def say(self, text: str) -> None:
        text = clean_for_speech(text, self.max_chars)
        sentences = split_sentences(text)

        if not sentences:
            return

        synthesizer = self.synthesizers.get(detect_language(text), self.synthesizer)

        with self._lock:
            self._stop.clear()
            audio: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue(maxsize=2)
            failures: list[Exception] = []

            def put(item) -> None:
                while not self._stop.is_set():
                    try:
                        audio.put(item, timeout=0.1)
                        return
                    except queue.Full:
                        continue

            def produce() -> None:
                try:
                    for sentence in sentences:
                        if self._stop.is_set():
                            return

                        put(synthesizer.synthesize(sentence))
                except Exception as e:
                    failures.append(e)
                finally:
                    put(None)

            worker = threading.Thread(target=produce, daemon=True)
            worker.start()

            try:
                while True:
                    try:
                        item = audio.get(timeout=0.1)
                    except queue.Empty:
                        if self._stop.is_set():
                            break

                        continue

                    if item is None:
                        break

                    if self._stop.is_set():
                        break

                    samples, rate = item

                    if len(samples):
                        self.player.play(samples, rate, self._stop.is_set)
            finally:
                self._stop.set()
                worker.join(timeout=5)

            if failures:
                raise SpeechError(f"speech synthesis failed: {failures[0]}") from failures[0]

    def beep(self) -> None:
        with self._lock:
            self.player.play(make_beep(), 16000)

    def stop(self) -> None:
        """Interrupt speech that is being synthesized or played."""
        self._stop.set()

    def close(self) -> None:
        self.stop()
        self.player.close()
