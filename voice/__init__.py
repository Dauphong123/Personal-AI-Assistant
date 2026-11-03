"""Wake-word voice interface. Heavy audio/ML libraries are imported only when voice mode starts."""

import threading
from collections.abc import Callable


def build_assistant(config, chat, show: Callable[[str], None] = print):
    """Create a VoiceAssistant with the real engines (needs requirements_voice.txt)."""
    try:
        from voice.assistant import VoiceAssistant
        from voice.audio import AudioPlayer, MicStream
        from voice.stt import FasterWhisperTranscriber
        from voice.tts import LazySynthesizer, PiperSynthesizer, Speaker
        from voice.vad import CommandRecorder, SileroVAD
        from voice.wakeword import OpenWakeWordEngine, WakeWordDetector
    except ImportError as e:
        raise RuntimeError(
            f"Voice mode needs extra packages ({e.name}). "
            "Install them with: python -m pip install -r requirements_voice.txt"
        ) from e

    if config.speaker_verification_enabled and not config.voiceprint_path.is_file():
        raise RuntimeError(
            "Speaker verification is enabled but there is no voiceprint. "
            "Run `python -m voice.enroll`, or set SPEAKER_VERIFICATION=false."
        )

    stop = threading.Event()
    show("Loading voice models (the first run downloads them)...")

    detector = WakeWordDetector(
        OpenWakeWordEngine(config.wake_word),
        threshold=config.wake_word_threshold,
        min_frames=config.wake_word_min_frames,
        cooldown=config.wake_word_cooldown,
    )
    recorder = CommandRecorder(
        SileroVAD(config.vad_threshold),
        end_silence=config.end_silence,
        should_stop=stop.is_set,
    )
    # Voices load on first use; the one for your own language is loaded now so the first reply is not delayed.
    voices = {
        "en": LazySynthesizer(lambda: PiperSynthesizer(config)),
        "vi": LazySynthesizer(lambda: PiperSynthesizer(config, config.tts_voice_vi)),
    }
    voices["vi" if config.stt_languages[:1] == ("vi",) else "en"].load()
    speaker = Speaker(
        voices["en"],
        AudioPlayer(config.output_device),
        max_chars=config.max_spoken_chars,
        synthesizers={"vi": voices["vi"]},
    )

    verifier = None

    if config.speaker_verification_enabled:
        from voice.speaker import SpeakerEmbedder, SpeakerVerifier, VoicePrint

        verifier = SpeakerVerifier(
            SpeakerEmbedder(config.model_dir),
            VoicePrint.load(config.voiceprint_path),
            config.speaker_threshold,
        )
        show(f"Speaker verification on (threshold {verifier.threshold:.2f}).")

    return VoiceAssistant(
        chat,
        MicStream(config.microphone_device),
        detector,
        recorder,
        FasterWhisperTranscriber(config),
        speaker,
        config,
        show=show,
        stop_event=stop,
        verifier=verifier,
    )
