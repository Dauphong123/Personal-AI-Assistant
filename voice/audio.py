"""Microphone capture and speaker playback (PortAudio via sounddevice)."""

import logging
import queue
from collections.abc import Callable

import numpy as np

from voice.wakeword import FRAME_SAMPLES, SAMPLE_RATE

logger = logging.getLogger(__name__)


class MicDisconnected(RuntimeError):
    """The microphone stream stopped delivering audio."""


class MicStream:
    """Captures 80 ms mono int16 frames at 16 kHz into a bounded queue.

    The PortAudio callback only copies the frame into the queue; all processing
    happens on the consumer's thread. If the consumer falls behind, frames are dropped.
    """

    def __init__(self, device: int | str | None = None, max_frames: int = 250):
        self.device = device
        self.dropped = 0
        self._frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_frames)
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        self.close()
        self._frames = queue.Queue(maxsize=self._frames.maxsize)
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.debug("Audio input status: %s", status)

        try:
            self._frames.put_nowait(indata[:, 0].copy())
        except queue.Full:
            self.dropped += 1

    def read(self, timeout: float = 0.1) -> np.ndarray | None:
        """Next frame, or None if none arrived within `timeout`. Raises MicDisconnected."""
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            if self._stream is None or not self._stream.active:
                raise MicDisconnected("microphone stream is not running") from None

            return None

    def flush(self) -> None:
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        stream, self._stream = self._stream, None

        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.warning("Closing the microphone failed", exc_info=True)


class AudioPlayer:
    """Plays int16 mono audio on the output device, one clip at a time.

    The output stream stays open between clips (opening one takes a noticeable
    fraction of a second, which would put gaps between sentences) until `close()`,
    an interruption, or an error.
    """

    def __init__(self, device: int | str | None = None):
        self.device = device
        self._stream = None
        self._rate = 0

    def _open(self, sample_rate: int):
        import sounddevice as sd

        if self._stream is None or self._rate != sample_rate:
            self.close()
            self._stream = sd.OutputStream(
                samplerate=sample_rate, channels=1, dtype="int16", device=self.device
            )
            self._stream.start()
            self._rate = sample_rate

        return self._stream

    def play(
        self,
        samples: np.ndarray,
        sample_rate: int,
        should_stop: Callable[[], bool] = lambda: False,
    ) -> None:
        stream = self._open(sample_rate)
        chunk = sample_rate // 10

        try:
            for start in range(0, len(samples), chunk):
                if should_stop():
                    self.close(drain=False)  # drop what is already buffered
                    return

                stream.write(np.ascontiguousarray(samples[start : start + chunk]).reshape(-1, 1))
        except Exception:
            self.close(drain=False)
            raise

    def close(self, drain: bool = True) -> None:
        """Close the output. By default the audio still buffered is played out first; closing a
        PortAudio stream directly would cut off the end of the last sentence."""
        stream, self._stream = self._stream, None

        if stream is not None:
            try:
                if drain:
                    stream.stop()  # returns when everything buffered has been played
                else:
                    stream.abort()

                stream.close()
            except Exception:
                logger.warning("Closing the audio output failed", exc_info=True)


def make_beep(sample_rate: int = SAMPLE_RATE, frequency: float = 880.0, seconds: float = 0.12) -> np.ndarray:
    """A short, soft activation tone."""
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    envelope = np.minimum(1.0, np.minimum(t, seconds - t) / 0.02)  # fade in/out: no click
    return (np.sin(2 * np.pi * frequency * t) * envelope * 0.3 * 32767).astype(np.int16)
