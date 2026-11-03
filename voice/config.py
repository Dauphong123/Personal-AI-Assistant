"""Voice settings, read from environment variables (the project's config mechanism)."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

TRUE = {"1", "true", "yes", "on"}
FALSE = {"0", "false", "no", "off", ""}


class VoiceConfigError(ValueError):
    pass


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)

    if raw is None:
        return default

    value = raw.strip().lower()

    if value in TRUE:
        return True

    if value in FALSE:
        return False

    raise VoiceConfigError(f"{name} must be true or false, got {raw!r}")


def _number(env: Mapping[str, str], name: str, default: float, low: float, high: float) -> float:
    raw = env.get(name)

    if raw is None or not raw.strip():
        return default

    try:
        value = float(raw)
    except ValueError:
        raise VoiceConfigError(f"{name} must be a number, got {raw!r}") from None

    if not low <= value <= high:
        raise VoiceConfigError(f"{name} must be between {low} and {high}, got {value}")

    return value


def _tri_bool(env: Mapping[str, str], name: str) -> bool | None:
    return None if not (env.get(name) or "").strip() else _bool(env, name, False)


def _optional_number(env: Mapping[str, str], name: str, low: float, high: float) -> float | None:
    return None if not (env.get(name) or "").strip() else _number(env, name, 0.0, low, high)


def _voiceprint_path(env: Mapping[str, str], default: "VoiceConfig") -> Path:
    model_dir = Path(_text(env, "VOICE_MODEL_DIR", str(default.model_dir))).expanduser()
    return Path(_text(env, "VOICEPRINT_PATH", str(model_dir / "voiceprint.npz"))).expanduser()


def _language(env: Mapping[str, str], default: "VoiceConfig") -> str:
    raw = _text(env, "SPEECH_LANGUAGE", default.stt_language).lower()
    codes = [code.strip() for code in raw.split(",")]

    for code in codes:
        if code != "auto" and not (code.isalpha() and 2 <= len(code) <= 3):
            raise VoiceConfigError(
                f'SPEECH_LANGUAGE must be language codes like "en", "vi" or "vi,en", or "auto"; got {raw!r}'
            )

    if "auto" in codes and len(codes) > 1:
        raise VoiceConfigError('SPEECH_LANGUAGE "auto" cannot be combined with other languages')

    return ",".join(codes)


def _default_stt_model(env: Mapping[str, str], default: "VoiceConfig") -> str:
    """Whisper's small model for anything but English: the base model garbles Vietnamese."""
    return default.stt_model if _language(env, default) == "en" else "small"


def _device(env: Mapping[str, str], name: str) -> int | str | None:
    raw = (env.get(name) or "").strip()

    if not raw:
        return None

    return int(raw) if raw.isdigit() else raw


def _text(env: Mapping[str, str], name: str, default: str) -> str:
    return (env.get(name) or "").strip() or default


@dataclass(frozen=True)
class VoiceConfig:
    enabled: bool = False

    # Wake word: a built-in openWakeWord model name, or a path to a .onnx model.
    wake_word: str = "hey_jarvis"
    wake_word_threshold: float = 0.5
    wake_word_min_frames: int = 2  # consecutive 80 ms frames above the threshold
    wake_word_cooldown: float = 2.0  # seconds ignored after a detection

    # Language(s) you speak: "en", "vi", a comma list such as "vi,en" (detect which one each time,
    # among just those), or "auto" (detect among all languages).
    stt_language: str = "en"
    stt_model: str = "base"  # for other languages from_env picks "small", which is much more accurate
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"

    tts_voice: str = "en_US-lessac-medium"
    tts_voice_vi: str = "vi_VN-vais1000-medium"  # used for replies written in Vietnamese
    max_spoken_chars: int = 1200

    listening_timeout: float = 10.0  # waiting for the user to start speaking
    max_recording_duration: float = 60.0
    end_silence: float = 0.8  # silence that ends a command
    vad_threshold: float = 0.5

    # Speaker verification: None = on when a voiceprint exists (see voice.enroll).
    speaker_verification: bool | None = None
    speaker_threshold: float | None = None  # None = the threshold calibrated at enrollment
    voiceprint_path: Path = Path.home() / ".cache" / "langchain-voice" / "voiceprint.npz"

    microphone_device: int | str | None = None
    output_device: int | str | None = None
    model_dir: Path = Path.home() / ".cache" / "langchain-voice"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "VoiceConfig":
        env = os.environ if env is None else env
        default = cls()

        config = cls(
            enabled=_bool(env, "VOICE_ENABLED", default.enabled),
            wake_word=_text(env, "WAKE_WORD", default.wake_word),
            wake_word_threshold=_number(env, "WAKE_WORD_THRESHOLD", default.wake_word_threshold, 0.05, 1.0),
            wake_word_min_frames=int(_number(env, "WAKE_WORD_MIN_FRAMES", default.wake_word_min_frames, 1, 10)),
            wake_word_cooldown=_number(env, "WAKE_WORD_COOLDOWN", default.wake_word_cooldown, 0, 30),
            stt_model=_text(env, "SPEECH_RECOGNITION_MODEL", _default_stt_model(env, default)),
            stt_language=_language(env, default),
            stt_device=_text(env, "SPEECH_RECOGNITION_DEVICE", default.stt_device),
            stt_compute_type=_text(env, "SPEECH_RECOGNITION_COMPUTE_TYPE", default.stt_compute_type),
            tts_voice=_text(env, "TTS_VOICE", default.tts_voice),
            tts_voice_vi=_text(env, "TTS_VOICE_VI", default.tts_voice_vi),
            max_spoken_chars=int(_number(env, "VOICE_MAX_SPOKEN_CHARS", default.max_spoken_chars, 50, 20000)),
            listening_timeout=_number(env, "VOICE_LISTENING_TIMEOUT", default.listening_timeout, 1, 120),
            max_recording_duration=_number(env, "VOICE_MAX_RECORDING_DURATION", default.max_recording_duration, 2, 600),
            end_silence=_number(env, "VOICE_END_SILENCE", default.end_silence, 0.2, 5),
            vad_threshold=_number(env, "VOICE_VAD_THRESHOLD", default.vad_threshold, 0.05, 0.95),
            speaker_verification=_tri_bool(env, "SPEAKER_VERIFICATION"),
            speaker_threshold=_optional_number(env, "SPEAKER_THRESHOLD", 0.1, 0.95),
            voiceprint_path=_voiceprint_path(env, default),
            microphone_device=_device(env, "MICROPHONE_DEVICE"),
            output_device=_device(env, "AUDIO_OUTPUT_DEVICE"),
            model_dir=Path(_text(env, "VOICE_MODEL_DIR", str(default.model_dir))).expanduser(),
        )

        if config.wake_word.lower().endswith(".onnx") and not Path(config.wake_word).expanduser().is_file():
            raise VoiceConfigError(f"WAKE_WORD model file not found: {config.wake_word}")

        return config

    @property
    def stt_languages(self) -> tuple[str, ...]:
        """The languages you speak, main one first; empty means detect among all of them."""
        return () if self.stt_language == "auto" else tuple(self.stt_language.split(","))

    @property
    def speaker_verification_enabled(self) -> bool:
        if self.speaker_verification is None:
            return self.voiceprint_path.is_file()

        return self.speaker_verification
