"""Wake-word detection: an engine scores audio frames; the detector decides when to fire."""

import logging
import time
from collections.abc import Callable
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

FRAME_SAMPLES = 1280  # 80 ms at 16 kHz, what openWakeWord expects
SAMPLE_RATE = 16000


class WakeWordEngine(Protocol):
    def score(self, frame: np.ndarray) -> float: ...

    def reset(self) -> None: ...


class WakeWordDetector:
    """Fires once per utterance: needs `min_frames` consecutive frames over the
    threshold, then ignores audio for `cooldown` seconds."""

    def __init__(
        self,
        engine: WakeWordEngine,
        threshold: float = 0.5,
        min_frames: int = 2,
        cooldown: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.engine = engine
        self.threshold = threshold
        self.min_frames = min_frames
        self.cooldown = cooldown
        self._clock = clock
        self._hits = 0
        self._muted_until = 0.0

    def process(self, frame: np.ndarray) -> bool:
        score = self.engine.score(frame)

        if self._clock() < self._muted_until:
            self._hits = 0
            return False

        if score < self.threshold:
            self._hits = 0
            return False

        self._hits += 1

        if self._hits < self.min_frames:
            return False

        self._hits = 0
        self._muted_until = self._clock() + self.cooldown
        self.engine.reset()
        logger.info("Wake word detected (score %.2f)", score)
        return True

    def reset(self) -> None:
        """Forget buffered audio, e.g. after the assistant spoke."""
        self._hits = 0
        self.engine.reset()


class OpenWakeWordEngine:
    """openWakeWord on the ONNX backend (the tflite backend is not available on Windows)."""

    def __init__(self, wake_word: str):
        import openwakeword
        from openwakeword.model import Model

        self._model = self._load(openwakeword, Model, wake_word)

    @staticmethod
    def _load(openwakeword, Model, wake_word: str):
        try:
            return Model(wakeword_models=[wake_word], inference_framework="onnx")
        except Exception:
            # Built-in models are downloaded on first use.
            logger.info("Downloading openWakeWord models")
            openwakeword.utils.download_models()
            return Model(wakeword_models=[wake_word], inference_framework="onnx")

    def score(self, frame: np.ndarray) -> float:
        scores = self._model.predict(frame)
        return float(max(scores.values(), default=0.0))

    def reset(self) -> None:
        self._model.reset()
