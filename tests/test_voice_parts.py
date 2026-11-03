"""Unit tests for the individual voice components (no audio hardware, no models)."""

import threading
import unittest

import numpy as np

from tests import stubs

stubs.install()

from voice import text as vtext  # noqa: E402
from voice.tts import Speaker, SpeechError  # noqa: E402
from voice.vad import FRAME_SECONDS, CommandRecorder  # noqa: E402
from voice.wakeword import FRAME_SAMPLES, WakeWordDetector  # noqa: E402

LOUD = np.ones(FRAME_SAMPLES, dtype=np.int16)
QUIET = np.zeros(FRAME_SAMPLES, dtype=np.int16)


class ScriptedEngine:
    def __init__(self, scores):
        self.scores = list(scores)
        self.resets = 0

    def score(self, frame):
        return self.scores.pop(0) if self.scores else 0.0

    def reset(self):
        self.resets += 1


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class WakeWordDetectorTests(unittest.TestCase):
    def detect(self, scores, **kwargs):
        clock = Clock()
        detector = WakeWordDetector(ScriptedEngine(scores), clock=clock, **kwargs)
        fired = []

        for i in range(len(scores)):
            clock.now = i * 0.08

            if detector.process(QUIET):
                fired.append(i)

        return fired

    def test_fires_when_score_stays_over_threshold(self):
        self.assertEqual(self.detect([0.1, 0.9, 0.9, 0.1], threshold=0.5, min_frames=2), [2])

    def test_single_noisy_frame_is_a_false_detection_and_ignored(self):
        self.assertEqual(self.detect([0.1, 0.95, 0.1, 0.1], threshold=0.5, min_frames=2), [])

    def test_sensitivity_threshold(self):
        scores = [0.6] * 4
        self.assertEqual(self.detect(scores, threshold=0.5, min_frames=1, cooldown=0), [0, 1, 2, 3])
        self.assertEqual(self.detect(scores, threshold=0.8, min_frames=1), [])

    def test_one_utterance_activates_once(self):
        # Scores stay high for ~1 s after a single "Hey Jarvis".
        self.assertEqual(self.detect([0.9] * 12, threshold=0.5, min_frames=2, cooldown=2.0), [1])

    def test_fires_again_after_the_cooldown(self):
        clock = Clock()
        detector = WakeWordDetector(ScriptedEngine([0.9] * 6), min_frames=1, cooldown=1.0, clock=clock)
        self.assertTrue(detector.process(QUIET))
        clock.now = 0.5
        self.assertFalse(detector.process(QUIET))
        clock.now = 1.5
        self.assertTrue(detector.process(QUIET))

    def test_reset_clears_engine_state(self):
        engine = ScriptedEngine([])
        WakeWordDetector(engine).reset()
        self.assertEqual(engine.resets, 1)


class EnergyVAD:
    """Speech = frames whose first sample is non-zero."""

    def __init__(self):
        self.resets = 0

    def is_speech(self, frame):
        return bool(frame[0])

    def reset(self):
        self.resets += 1


class ScriptedMic:
    def __init__(self, frames):
        self.frames = list(frames)

    def read(self, timeout=0.1):
        return self.frames.pop(0) if self.frames else None


class CommandRecorderTests(unittest.TestCase):
    def recorder(self, **kwargs):
        return CommandRecorder(EnergyVAD(), end_silence=0.35, **kwargs)

    def test_records_until_silence_after_speech(self):
        mic = ScriptedMic([QUIET, LOUD, LOUD, *[QUIET] * 6, LOUD])
        audio = self.recorder().record(mic)
        self.assertIsNotNone(audio)
        # 1 lead-in frame + 2 speech + 5 silent frames (0.4 s >= 0.35 s); the later speech is left alone
        self.assertEqual(len(audio) // FRAME_SAMPLES, 8)
        self.assertEqual(len(mic.frames), 2)

    def test_no_speech_times_out(self):
        recorder = self.recorder()
        self.assertIsNone(recorder.record(ScriptedMic([QUIET] * 100), listen_timeout=1.0))

    def test_preroll_speech_is_kept(self):
        recorder = self.recorder()
        audio = recorder.record(ScriptedMic([LOUD, QUIET, QUIET, QUIET, QUIET, QUIET]), preroll=[LOUD, LOUD])
        self.assertGreaterEqual(int(np.count_nonzero(audio[::FRAME_SAMPLES])), 3)

    def test_max_duration_ends_a_long_recording(self):
        recorder = self.recorder()
        frames = [LOUD] * 100
        audio = recorder.record(ScriptedMic(frames), max_duration=1.0)
        self.assertLessEqual(len(audio) / 16000, 1.0 + 2 * FRAME_SECONDS)

    def test_stop_request_aborts_listening(self):
        stop = threading.Event()
        stop.set()
        recorder = CommandRecorder(EnergyVAD(), should_stop=stop.is_set)
        self.assertIsNone(recorder.record(ScriptedMic([])))

    def test_stalled_stream_after_speech_returns_what_was_heard(self):
        audio = self.recorder().record(ScriptedMic([LOUD, LOUD]))
        self.assertEqual(len(audio) // FRAME_SAMPLES, 2)


class FakeSynth:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def synthesize(self, text):
        self.calls.append(text)

        if text == self.fail_on:
            raise RuntimeError("synth broke")

        return np.full(100, len(text), dtype=np.int16), 22050


class FakePlayer:
    def __init__(self):
        self.played = []
        self.closed = 0
        self.active = 0
        self.max_active = 0
        self.during_play = None  # callback run in the middle of the first clip

    def play(self, samples, rate, should_stop=lambda: False):
        self.active += 1
        self.max_active = max(self.max_active, self.active)

        if self.during_play and not self.played:
            self.during_play()

        if not should_stop():
            self.played.append((int(samples[0]), rate))

        self.active -= 1

    def close(self):
        self.closed += 1


class SpeakerTests(unittest.TestCase):
    def test_reply_is_spoken_sentence_by_sentence_in_order(self):
        synth, player = FakeSynth(), FakePlayer()
        Speaker(synth, player).say("First one. Second one! Third?")
        self.assertEqual(synth.calls, ["First one.", "Second one!", "Third?"])
        self.assertEqual(player.played, [(10, 22050), (11, 22050), (6, 22050)])

    def test_markdown_code_and_thinking_are_not_read_aloud(self):
        synth = FakeSynth()
        Speaker(synth, FakePlayer()).say("<think>secret plan</think>**Done.** See `x.py`.\n```\nprint(1)\n```")
        spoken = " ".join(synth.calls)
        self.assertNotIn("secret", spoken)
        self.assertNotIn("print", spoken)
        self.assertNotIn("*", spoken)
        self.assertIn("Done.", spoken)

    def test_nothing_to_say_plays_nothing(self):
        player = FakePlayer()
        Speaker(FakeSynth(), player).say("<think>only thoughts</think>")
        self.assertEqual(player.played, [])

    def test_synthesis_failure_raises_speech_error_after_earlier_sentences_played(self):
        player = FakePlayer()
        speaker = Speaker(FakeSynth(fail_on="Bad one."), player)

        with self.assertRaises(SpeechError):
            speaker.say("Good one. Bad one. Never reached.")

        self.assertEqual(len(player.played), 1)

    def test_concurrent_say_calls_never_overlap(self):
        player = FakePlayer()
        speaker = Speaker(FakeSynth(), player)
        threads = [threading.Thread(target=speaker.say, args=(f"Message {i}. Again {i}.",)) for i in range(4)]
        [t.start() for t in threads]
        [t.join(5) for t in threads]
        self.assertEqual(len(player.played), 8)
        self.assertEqual(player.max_active, 1)

    def test_stop_during_playback_skips_the_remaining_sentences(self):
        synth, player = FakeSynth(), FakePlayer()
        speaker = Speaker(synth, player)
        player.during_play = speaker.stop  # the user interrupts while the first sentence plays
        speaker.say("One. Two. Three. Four. Five.")
        self.assertEqual(player.played, [])
        self.assertLessEqual(len(synth.calls), 3)

    def test_speaking_works_again_after_a_stop(self):
        player = FakePlayer()
        speaker = Speaker(FakeSynth(), player)
        speaker.stop()
        speaker.say("Hello again.")
        self.assertEqual(len(player.played), 1)

    def test_close_releases_the_player(self):
        player = FakePlayer()
        Speaker(FakeSynth(), player).close()
        self.assertEqual(player.closed, 1)

    def test_beep_uses_the_player(self):
        player = FakePlayer()
        Speaker(FakeSynth(), player).beep()
        self.assertEqual(len(player.played), 1)


class TextTests(unittest.TestCase):
    def test_wake_word_phrases(self):
        self.assertEqual(vtext.wake_word_phrases("hey_jarvis"), ["hey jarvis", "jarvis"])
        self.assertEqual(vtext.wake_word_phrases("C:/m/hey_jarvis_v0.1.onnx"), ["hey jarvis", "jarvis"])
        self.assertEqual(vtext.wake_word_phrases("jarvis"), ["jarvis"])

    def test_strip_wake_word(self):
        cases = {
            "Hey Jarvis, what time is it?": "what time is it?",
            "Jarvis, explain my project.": "explain my project.",
            "jarvis": "",
            "Hey, Jarvis! open it": "open it",
            "What does Jarvis mean?": "What does Jarvis mean?",
            "Travis is here": "Travis is here",
        }
        for heard, expected in cases.items():
            self.assertEqual(vtext.strip_wake_word(heard, "hey_jarvis"), expected, heard)

    def test_recognizer_noise_is_filler_but_requests_are_not(self):
        for noise in ["you", "You.", "Uh...", "thank you", "Hmm"]:
            self.assertTrue(vtext.is_filler(noise), noise)

        for request in ["you are slow today", "stop", "what time is it", "thank you for the help"]:
            self.assertFalse(vtext.is_filler(request), request)

    def test_normalize_answer(self):
        for yes in ["Yes.", "yeah", "Okay!", "yes please", "go ahead"]:
            self.assertEqual(vtext.normalize_answer(yes), "y", yes)

        for no in ["no", "hmm", "I don't think so", "not now", ""]:
            self.assertEqual(vtext.normalize_answer(no), "n", no)

        self.assertEqual(vtext.normalize_answer("stop"), "stop")

    def test_long_replies_are_trimmed_at_a_sentence(self):
        reply = "This is a sentence. " * 100
        spoken = vtext.clean_for_speech(reply, 200)
        self.assertLess(len(spoken), 260)
        self.assertIn("The rest is on screen.", spoken)

    def test_horizontal_rules_and_symbol_only_pieces_are_not_spoken(self):
        spoken = vtext.clean_for_speech("Intro.\n\n---\n\nNext part.", 500)
        self.assertEqual(vtext.split_sentences(spoken), ["Intro.", "Next part."])
        self.assertEqual(vtext.split_sentences("Hello there. ... ?! Next one."), ["Hello there.", "Next one."])


if __name__ == "__main__":
    unittest.main()
