"""Voice activity detection and command recording (finding where speech starts and ends)."""

import logging
from collections.abc import Callable
from typing import Protocol

import numpy as np

from voice.wakeword import FRAME_SAMPLES, SAMPLE_RATE

logger = logging.getLogger(__name__)

FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE
VAD_WINDOW = 512  # samples Silero VAD accepts per call at 16 kHz


class VoiceActivityDetector(Protocol):
    def is_speech(self, frame: np.ndarray) -> bool: ...

    def reset(self) -> None: ...


class SileroVAD:
    """Silero VAD on ONNX. Feeds the model 512-sample windows; a frame is speech if any window is."""

    def __init__(self, threshold: float = 0.5):
        import torch
        from silero_vad import load_silero_vad

        self._torch = torch
        self._model = load_silero_vad(onnx=True)
        self.threshold = threshold
        self._rest = np.zeros(0, dtype=np.int16)

    def is_speech(self, frame: np.ndarray) -> bool:
        samples = np.concatenate([self._rest, frame])
        speech = False
        usable = len(samples) - len(samples) % VAD_WINDOW

        for start in range(0, usable, VAD_WINDOW):
            window = samples[start : start + VAD_WINDOW].astype(np.float32) / 32768.0
            prob = float(self._model(self._torch.from_numpy(window), SAMPLE_RATE))
            speech = speech or prob >= self.threshold

        self._rest = samples[usable:]
        return speech

    def reset(self) -> None:
        self._model.reset_states()
        self._rest = np.zeros(0, dtype=np.int16)


class MicrophoneLike(Protocol):
    def read(self, timeout: float = ...) -> np.ndarray | None: ...


class CommandRecorder:
    """Records one utterance: waits for speech, then stops after `end_silence` of quiet."""

    def __init__(
        self,
        vad: VoiceActivityDetector,
        end_silence: float = 0.8,
        should_stop: Callable[[], bool] = lambda: False,
    ):
        self.vad = vad
        self.end_silence = end_silence
        self._should_stop = should_stop

    def record(
        self,
        mic: MicrophoneLike,
        preroll: list[np.ndarray] | None = None,
        listen_timeout: float = 10.0,
        max_duration: float = 60.0,
    ) -> np.ndarray | None:
        """Return the audio of one utterance, or None if nobody spoke before the timeout.

        `preroll` frames come first, so speech that began just before this call is kept."""
        self.vad.reset()
        frames: list[np.ndarray] = []
        waited = 0.0
        recorded = 0.0
        silence = 0.0
        started = False

        def feed(frame: np.ndarray) -> bool:
            """Handle one frame; True when the utterance is complete."""
            nonlocal started, silence, recorded, waited

            speech = self.vad.is_speech(frame)

            if not started:
                if not speech:
                    waited += FRAME_SECONDS
                    # Keep a little lead-in so the first syllable isn't clipped.
                    frames.append(frame)
                    del frames[:-4]
                    return False

                started = True

            frames.append(frame)
            recorded += FRAME_SECONDS
            silence = 0.0 if speech else silence + FRAME_SECONDS
            return silence >= self.end_silence or recorded >= max_duration

        for frame in preroll or []:
            if feed(frame):
                return np.concatenate(frames)

        stalled = 0.0

        while not self._should_stop():
            frame = mic.read(timeout=0.5)

            if frame is None:
                stalled += 0.5

                if started and stalled >= 2.0:
                    return np.concatenate(frames)  # the audio stream stopped delivering

                waited += 0.5 if not started else 0.0
            else:
                stalled = 0.0

                if feed(frame):
                    return np.concatenate(frames)

            if not started and waited >= listen_timeout:
                return None

        return None
