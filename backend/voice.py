"""Voice for the web chat: the browser records, the backend transcribes and speaks.

Push-to-talk replaces the wake word: the page uploads one recording (webm/ogg/wav, decoded with
PyAV through faster-whisper), gets the text back and sends it like a typed message. Answers are
cleaned for speech (voice.text) and synthesized with Piper into a WAV the page plays. Models
come from voice/ (settings in .env, see voice/CLAUDE.md) and load on first use.
"""

import importlib.util
import io
import logging
import threading
import wave
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)

SPEECH_LIMIT = 1200  # characters read aloud per answer (as the keyboard voice mode)
STT_PACKAGES = ("faster_whisper", "av")
TTS_PACKAGES = ("piper",)


def decode_recording(data: bytes, rate: int = 16000) -> np.ndarray:
    """Browser audio (any container PyAV reads: webm/opus, ogg, mp4, wav) as mono float32 in
    [-1, 1] at `rate`. PyAV is used directly: faster_whisper.decode_audio (1.2.1) passes
    av.open() an argument PyAV 19 no longer accepts."""
    import av

    resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
    chunks: list[np.ndarray] = []

    with av.open(io.BytesIO(data), mode="r") as container:
        if not container.streams.audio:
            raise ValueError("The recording has no audio track")

        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))

        for out in resampler.resample(None):  # flush
            chunks.append(out.to_ndarray().reshape(-1))

    if not chunks:
        return np.zeros(0, dtype=np.float32)

    return np.concatenate(chunks).astype(np.float32) / 32768.0


class WebVoice:
    def __init__(
        self,
        transcriber: Callable[[], object] | None = None,
        synthesizer: Callable[[str], object] | None = None,
        decode: Callable[[bytes], np.ndarray] = decode_recording,
        speech_limit: int = SPEECH_LIMIT,
    ):
        """`transcriber()` builds an object with transcribe(int16 array) -> str; `synthesizer(language)`
        one with synthesize(text) -> (int16 samples, rate). None means that half is unavailable."""
        self._make_transcriber = transcriber
        self._make_synthesizer = synthesizer
        self._decode = decode
        self._speech_limit = speech_limit
        self._transcriber = None
        self._synthesizers: dict[str, object] = {}
        self._stt_lock = threading.Lock()  # one recording at a time per model
        self._tts_lock = threading.Lock()

    @property
    def can_transcribe(self) -> bool:
        return self._make_transcriber is not None

    @property
    def can_speak(self) -> bool:
        return self._make_synthesizer is not None

    @classmethod
    def from_env(cls, find_spec: Callable[[str], object] = importlib.util.find_spec) -> "WebVoice":
        """Engines for the installed packages (requirements_voice.txt); nothing is loaded yet."""
        stt = all(find_spec(name) is not None for name in STT_PACKAGES)
        tts = all(find_spec(name) is not None for name in TTS_PACKAGES)

        def config():
            from voice.config import VoiceConfig

            return VoiceConfig.from_env()

        def transcriber():
            from voice.stt import FasterWhisperTranscriber

            return FasterWhisperTranscriber(config())

        def synthesizer(language: str):
            from voice.tts import PiperSynthesizer

            settings = config()
            return PiperSynthesizer(settings, settings.tts_voice_vi if language == "vi" else settings.tts_voice)

        return cls(transcriber if stt else None, synthesizer if tts else None)

    def transcribe(self, data: bytes) -> str:
        """Text of one browser recording; '' when there was no usable speech.
        Raises ValueError (via the decoder) for data that is not audio."""
        if self._make_transcriber is None:
            raise RuntimeError("Speech recognition is not available")

        samples = self._decode(data)
        audio = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)

        with self._stt_lock:
            if self._transcriber is None:
                self._transcriber = self._make_transcriber()

            return self._transcriber.transcribe(audio).strip()  # type: ignore[attr-defined]

    def speak(self, text: str) -> bytes:
        """A WAV file reading `text` aloud, or b'' when nothing in it is worth saying."""
        if self._make_synthesizer is None:
            raise RuntimeError("Speech output is not available")

        from language import detect_language
        from voice.text import clean_for_speech

        spoken = clean_for_speech(text, self._speech_limit)

        if not spoken:
            return b""

        language = "vi" if detect_language(spoken) == "vi" else "en"

        with self._tts_lock:
            if language not in self._synthesizers:
                self._synthesizers[language] = self._make_synthesizer(language)

            samples, rate = self._synthesizers[language].synthesize(spoken)  # type: ignore[attr-defined]

        out = io.BytesIO()

        with wave.open(out, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(int(rate))
            wav.writeframes(np.asarray(samples, dtype=np.int16).tobytes())

        return out.getvalue()
