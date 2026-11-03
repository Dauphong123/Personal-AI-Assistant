"""Voice enrollment: `python -m voice.enroll` (create a voiceprint) or `--test` (try it live).

You read a few fixed phrases; their speaker embeddings become your voiceprint. With a
voiceprint present, voice mode only accepts commands spoken in your voice.
"""

import sys
from collections.abc import Callable

import numpy as np

from voice.speaker import SAMPLE_RATE, Embedder, SpeakerVerifier, VoicePrint

PHRASES = [
    "Hey Jarvis, what is the weather like today?",
    "Please open the project folder and show me the latest changes.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Remind me to review the pull request before lunch.",
    "Hey Jarvis, explain how the memory system stores facts.",
    "Yes, allow it. No, stop that and cancel the task.",
]
PHRASES_VI = [
    "Này Jarvis, hôm nay thời tiết như thế nào?",
    "Hãy mở thư mục dự án và cho tôi xem các thay đổi mới nhất.",
    "Con cáo nâu nhanh nhẹn nhảy qua con chó lười bên bờ sông.",
    "Nhắc tôi xem lại yêu cầu hợp nhất trước bữa trưa.",
    "Này Jarvis, giải thích cho tôi cách hệ thống bộ nhớ lưu trữ thông tin.",
    "Có, cho phép. Không, dừng lại và hủy nhiệm vụ.",
]


def phrases_for(language: str) -> list[str]:
    """Read the phrases in the language you speak to the assistant."""
    return PHRASES_VI if language == "vi" else PHRASES


MIN_PHRASE_SECONDS = 1.5
ATTEMPTS_PER_PHRASE = 3


class EnrollmentError(RuntimeError):
    pass


def run_enrollment(
    mic,
    recorder,
    embedder: Embedder,
    phrases: list[str] = PHRASES,
    show: Callable[[str], None] = print,
) -> VoicePrint:
    """Record each phrase (retrying unusable takes) and build the voiceprint."""
    embeddings = []

    for number, phrase in enumerate(phrases, 1):
        for _ in range(ATTEMPTS_PER_PHRASE):
            show(f'\n[{number}/{len(phrases)}] Please say:  "{phrase}"')
            mic.flush()
            audio = recorder.record(mic, listen_timeout=15.0, max_duration=15.0)

            if audio is None:
                show("  I didn't hear anything. Let's try that phrase again.")
                continue

            seconds = len(audio) / SAMPLE_RATE

            if seconds < MIN_PHRASE_SECONDS:
                show(f"  That was too short ({seconds:.1f} s). Please say the whole phrase.")
                continue

            embeddings.append(embedder.embed(audio))
            show(f"  Got it ({seconds:.1f} s).")
            break
        else:
            raise EnrollmentError(f'Could not record a usable take of "{phrase}".')

    return VoicePrint.from_embeddings(embeddings)


def run_test(mic, recorder, verifier: SpeakerVerifier, rounds: int = 6, show=print) -> list[float]:
    """Speak freely (or have someone else speak) and see the similarity scores."""
    scores = []

    for round_ in range(1, rounds + 1):
        show(f"\n[{round_}/{rounds}] Say anything...")
        mic.flush()
        audio = recorder.record(mic, listen_timeout=15.0, max_duration=15.0)

        if audio is None:
            show("  (nothing heard)")
            continue

        score = verifier.score(audio) if len(audio) >= SAMPLE_RATE // 2 else float("nan")
        scores.append(score)
        verdict = "ACCEPTED" if score >= verifier.threshold else "rejected"
        show(f"  similarity {score:.2f}  (threshold {verifier.threshold:.2f})  -> {verdict}")

    return scores


def main(argv: list[str]) -> int:
    from dotenv import load_dotenv

    from voice.audio import MicStream
    from voice.config import VoiceConfig
    from voice.speaker import SpeakerEmbedder
    from voice.vad import CommandRecorder, SileroVAD

    load_dotenv()
    config = VoiceConfig.from_env()
    testing = "--test" in argv

    if testing and not config.voiceprint_path.is_file():
        print("No voiceprint yet. Run: python -m voice.enroll")
        return 1

    print("Loading models (the first run downloads a 26 MB speaker model)...")
    embedder = SpeakerEmbedder(config.model_dir)
    recorder = CommandRecorder(SileroVAD(config.vad_threshold), end_silence=config.end_silence)
    mic = MicStream(config.microphone_device)
    mic.start()

    try:
        if testing:
            voiceprint = VoicePrint.load(config.voiceprint_path)
            verifier = SpeakerVerifier(embedder, voiceprint, config.speaker_threshold)
            run_test(mic, recorder, verifier)
            return 0

        print("\nVoice enrollment: read each phrase aloud in your normal voice and speaking distance.")
        print("Stay in the place where you will use the assistant; background noise is part of the profile.")

        try:
            voiceprint = run_enrollment(mic, recorder, embedder, phrases_for(config.stt_languages[0] if config.stt_languages else 'en'))
        except EnrollmentError as e:
            print(f"\nEnrollment failed: {e}")
            return 1

        voiceprint.save(config.voiceprint_path)
        low = min(voiceprint.consistency)
        print(f"\nSaved your voiceprint to {config.voiceprint_path}")
        print(f"Consistency between your phrases: lowest {low:.2f}, average {np.mean(voiceprint.consistency):.2f}")
        print(f"Acceptance threshold: {voiceprint.threshold:.2f}")

        if low < 0.5:
            print("Some phrases did not sound alike (noise, distance, or mumbling). Consider enrolling again.")

        print("\nTry it: python -m voice.enroll --test")
        print("Voice mode now only accepts your voice. Set SPEAKER_VERIFICATION=false to turn that off.")
        return 0
    finally:
        mic.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
