"""Microphone and wake-word diagnostic: `python -m voice.check [device] [seconds]`.

Shows the input devices, then a live level meter and the wake-word score for the
chosen microphone (default: MICROPHONE_DEVICE, else the system default). Say the
wake word while it runs.
"""

import sys
import time

import numpy as np


def list_inputs() -> None:
    import sounddevice as sd

    default = sd.default.device[0]
    print("Input devices (use the number as MICROPHONE_DEVICE):")

    for index, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] < 1:
            continue

        api = sd.query_hostapis(dev["hostapi"])["name"]
        mark = "  <- default" if index == default else ""
        print(f"  [{index:2}] {dev['name']} ({api}){mark}")


def main(argv: list[str]) -> int:
    from dotenv import load_dotenv

    from voice.audio import MicDisconnected, MicStream
    from voice.config import VoiceConfig
    from voice.wakeword import OpenWakeWordEngine

    load_dotenv()
    config = VoiceConfig.from_env()
    device = config.microphone_device
    seconds = 12.0

    if argv and argv[0].strip():
        device = int(argv[0]) if argv[0].isdigit() else argv[0].strip()

    if len(argv) > 1:
        seconds = float(argv[1])

    list_inputs()
    print(f"\nUsing: {device if device is not None else 'system default'}   wake word: {config.wake_word}"
          f"   threshold: {config.wake_word_threshold}")
    engine = OpenWakeWordEngine(config.wake_word)
    mic = MicStream(device)
    mic.start()

    print(f"Say the wake word now ({seconds:.0f} s)...\n")
    best_score = 0.0
    loudest = 0.0
    end = time.monotonic() + seconds
    shown = 0.0

    try:
        while time.monotonic() < end:
            frame = mic.read(timeout=0.5)

            if frame is None:
                continue

            rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
            score = engine.score(frame)
            best_score = max(best_score, score)
            loudest = max(loudest, rms)

            if time.monotonic() - shown >= 0.25:
                shown = time.monotonic()
                bar = "#" * min(40, int(rms / 100))
                hit = "  <== WAKE WORD" if score >= config.wake_word_threshold else ""
                print(f"level {rms:7.0f} {bar:<40} score {score:4.2f}{hit}")
    except MicDisconnected:
        print("\nThe microphone stream stopped delivering audio.")
        return 1
    finally:
        mic.close()

    print()

    if loudest < 20:
        print("RESULT: the microphone is silent. Wrong device, muted, or blocked by Windows privacy settings.")
    elif best_score >= config.wake_word_threshold:
        print(f"RESULT: working. The wake word scored {best_score:.2f}.")
    else:
        print(
            f"RESULT: the microphone hears sound (peak level {loudest:.0f}) but the wake word scored only "
            f"{best_score:.2f}. Say the full phrase ({config.wake_word.replace('_', ' ')}), speak closer, "
            "or lower WAKE_WORD_THRESHOLD."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
