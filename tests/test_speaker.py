"""Speaker verification and enrollment tests (no models, microphone, or network)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import stubs

stubs.install()

from tests.test_voice import QUIET, WAKE, VoiceTestCase  # noqa: E402
from voice import enroll as enroll_module  # noqa: E402
from voice.config import VoiceConfig  # noqa: E402
from voice.speaker import (  # noqa: E402
    MIN_SAMPLES,
    SpeakerVerifier,
    VoicePrint,
    calibrate_threshold,
)

DIM = 16


def clip(speaker: int, seconds: float = 2.0) -> np.ndarray:
    """Audio whose first sample says who is speaking (the fake embedder reads it)."""
    return np.full(int(16000 * seconds), speaker, dtype=np.int16)


class FakeEmbedder:
    """Each speaker is a direction; a little deterministic noise per clip length."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def embed(self, audio):
        self.calls.append(len(audio))

        if self.fail:
            raise RuntimeError("model crashed")

        vector = np.zeros(DIM, dtype=np.float32)
        vector[int(audio[0]) % DIM] = 1.0
        vector += np.random.default_rng(len(audio)).normal(0, 0.05, DIM).astype(np.float32)
        return vector / np.linalg.norm(vector)


def owner_voiceprint(embedder=None) -> VoicePrint:
    embedder = embedder or FakeEmbedder()
    return VoicePrint.from_embeddings([embedder.embed(clip(1, 1.0 + i)) for i in range(5)])


class VoicePrintTests(unittest.TestCase):
    def test_threshold_sits_between_consistency_and_impostor_level_and_is_clamped(self):
        self.assertAlmostEqual(calibrate_threshold([0.75, 0.75]), 0.55)
        self.assertAlmostEqual(calibrate_threshold([0.99]), 0.67)
        self.assertEqual(calibrate_threshold([0.1]), 0.4)  # clamped low
        self.assertEqual(calibrate_threshold([5.0]), 0.75)  # clamped high

    def test_needs_several_clips(self):
        with self.assertRaises(ValueError):
            VoicePrint.from_embeddings([np.ones(DIM)] * 2)

    def test_save_and_load_roundtrip(self):
        voiceprint = owner_voiceprint()

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "sub" / "voiceprint.npz"
            voiceprint.save(path)
            loaded = VoicePrint.load(path)

        np.testing.assert_allclose(loaded.embeddings, voiceprint.embeddings)
        self.assertAlmostEqual(loaded.threshold, voiceprint.threshold)
        self.assertEqual(len(loaded.consistency), 5)


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.embedder = FakeEmbedder()
        self.verifier = SpeakerVerifier(self.embedder, owner_voiceprint(self.embedder))

    def test_owner_is_accepted(self):
        result = self.verifier.verify(clip(1))
        self.assertTrue(result.accepted, result)
        self.assertGreater(result.score, self.verifier.threshold)

    def test_other_speaker_is_rejected(self):
        result = self.verifier.verify(clip(2))
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "voice not recognized")

    def test_too_little_audio_is_rejected_without_embedding(self):
        before = len(self.embedder.calls)
        result = self.verifier.verify(clip(1, 0.3))
        self.assertFalse(result.accepted)
        self.assertIn("too little", result.reason)
        self.assertEqual(len(self.embedder.calls), before)
        self.assertLess(len(clip(1, 0.3)), MIN_SAMPLES)

    def test_embedding_failure_fails_closed(self):
        verifier = SpeakerVerifier(FakeEmbedder(fail=True), owner_voiceprint())
        result = verifier.verify(clip(1))
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "voice check failed")

    def test_threshold_override(self):
        strict = SpeakerVerifier(self.embedder, self.verifier.voiceprint, threshold=0.999)
        self.assertFalse(strict.verify(clip(1)).accepted)

    def test_only_the_last_seconds_are_embedded(self):
        self.verifier.verify(clip(1, 60.0))
        self.assertEqual(self.embedder.calls[-1], 16000 * 20)


class ScriptedRecorder:
    def __init__(self, takes):
        self.takes = list(takes)
        self.calls = 0

    def record(self, mic, preroll=None, listen_timeout=10, max_duration=60):
        self.calls += 1
        return self.takes.pop(0) if self.takes else None


class FlushMic:
    def __init__(self):
        self.flushes = 0

    def flush(self):
        self.flushes += 1


class EnrollmentTests(unittest.TestCase):
    PHRASES = ["one two three", "four five six", "seven eight nine"]

    def enroll(self, takes, phrases=None):
        shown = []
        recorder = ScriptedRecorder(takes)
        voiceprint = enroll_module.run_enrollment(
            FlushMic(), recorder, FakeEmbedder(), phrases or self.PHRASES, show=shown.append
        )
        return voiceprint, recorder, "\n".join(shown)

    def test_one_clean_take_per_phrase_builds_a_voiceprint(self):
        voiceprint, recorder, shown = self.enroll([clip(1, 2.0), clip(1, 2.5), clip(1, 3.0)])
        self.assertEqual(len(voiceprint.embeddings), 3)
        self.assertEqual(recorder.calls, 3)
        self.assertIn('[1/3] Please say:  "one two three"', shown)

    def test_short_or_missing_takes_are_retried(self):
        takes = [clip(1, 0.5), None, clip(1, 2.0), clip(1, 2.0), clip(1, 2.0)]
        voiceprint, recorder, shown = self.enroll(takes)
        self.assertEqual(len(voiceprint.embeddings), 3)
        self.assertEqual(recorder.calls, 5)
        self.assertIn("too short", shown)
        self.assertIn("didn't hear anything", shown)

    def test_gives_up_after_repeated_bad_takes(self):
        with self.assertRaises(enroll_module.EnrollmentError):
            self.enroll([clip(1, 0.4)] * 3)

    def test_test_mode_reports_scores_and_verdicts(self):
        embedder = FakeEmbedder()
        verifier = SpeakerVerifier(embedder, owner_voiceprint(embedder))
        shown = []
        scores = enroll_module.run_test(
            FlushMic(), ScriptedRecorder([clip(1), clip(2)]), verifier, rounds=2, show=shown.append
        )
        self.assertGreater(scores[0], verifier.threshold)
        self.assertLess(scores[1], verifier.threshold)
        text = "\n".join(shown)
        self.assertIn("ACCEPTED", text)
        self.assertIn("rejected", text)


class ConfigTests(unittest.TestCase):
    def test_verification_is_automatic_when_a_voiceprint_exists(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "vp.npz"
            env = {"VOICEPRINT_PATH": str(path)}
            self.assertFalse(VoiceConfig.from_env(env).speaker_verification_enabled)
            path.write_bytes(b"x")
            self.assertTrue(VoiceConfig.from_env(env).speaker_verification_enabled)
            off = VoiceConfig.from_env({**env, "SPEAKER_VERIFICATION": "false"})
            self.assertFalse(off.speaker_verification_enabled)

    def test_explicit_on_and_threshold(self):
        config = VoiceConfig.from_env({"SPEAKER_VERIFICATION": "true", "SPEAKER_THRESHOLD": "0.6"})
        self.assertTrue(config.speaker_verification_enabled)
        self.assertEqual(config.speaker_threshold, 0.6)


class AssistantVoiceCheckTests(VoiceTestCase):
    def setUp(self):
        self.embedder = FakeEmbedder()
        self.verifier = SpeakerVerifier(self.embedder, owner_voiceprint(self.embedder))

    def test_owner_command_goes_through(self):
        self.build(frames=[WAKE], audios=[clip(1)], texts=["Jarvis, open the file"], verifier=self.verifier)
        self.run_until(lambda: self.chat.submitted)
        self.assertEqual(self.chat.submitted, ["open the file"])

    def test_stranger_is_rejected_and_nothing_reaches_the_agent(self):
        self.build(frames=[WAKE], audios=[clip(2)], texts=["Jarvis, delete everything"], verifier=self.verifier)
        self.run_until(lambda: self.assistant.cycles >= 1)
        self.assertEqual(self.chat.submitted, [])
        self.assertIn("[REJECTED]", self.text)
        self.assertNotIn("delete everything", self.text, "a rejected transcript must not be echoed")

    def test_wake_word_then_pause_then_command_is_judged_on_all_the_speech(self):
        # "Jarvis." alone is too short to judge (0.4 s); with the command it is enough.
        self.build(
            frames=[WAKE], audios=[clip(1, 0.4), clip(1, 0.5)],
            texts=["Jarvis.", "what time is it"], verifier=self.verifier,
        )  # fmt: skip
        self.run_until(lambda: self.chat.submitted)
        self.assertEqual(self.chat.submitted, ["what time is it"])
        self.assertEqual(self.embedder.calls[-1], int(16000 * 0.4) + int(16000 * 0.5))

    def test_too_short_command_is_rejected(self):
        self.build(frames=[WAKE], audios=[clip(1, 0.3)], texts=["Jarvis, stop"], verifier=self.verifier)
        self.run_until(lambda: self.assistant.cycles >= 1)
        self.assertEqual(self.chat.submitted, [])
        self.assertIn("Too little speech", self.text)

    def test_stranger_cannot_approve_a_tool(self):
        from tests.test_voice import FakeChat

        chat = FakeChat()
        chat.busy = True
        chat.pending_approval = {"calls": [{"name": "terminal", "args": {"command": "rm -rf x"}}]}
        self.build(audios=[clip(2)], texts=["yes allow it"], chat=chat, verifier=self.verifier)
        self.run_until(lambda: chat.submitted)
        self.assertEqual(chat.submitted, ["n"])

    def test_owner_can_approve_a_tool(self):
        from tests.test_voice import FakeChat

        chat = FakeChat()
        chat.busy = True
        chat.pending_approval = {"calls": [{"name": "terminal", "args": {"command": "ls"}}]}
        self.build(audios=[clip(1)], texts=["Yes, allow it."], chat=chat, verifier=self.verifier)
        self.run_until(lambda: chat.submitted)
        self.assertEqual(chat.submitted, ["y"])

    def test_without_a_verifier_any_voice_is_accepted(self):
        self.build(frames=[WAKE], audios=[clip(2)], texts=["Jarvis, hello"])
        self.run_until(lambda: self.chat.submitted)
        self.assertEqual(self.chat.submitted, ["hello"])

    def test_interaction_audio_does_not_leak_into_the_next_one(self):
        self.build(
            frames=[WAKE, WAKE], audios=[clip(1, 1.0), clip(1, 1.5)],
            texts=["Jarvis, one", "Jarvis, two"], verifier=self.verifier,
        )  # fmt: skip
        self.run_until(lambda: len(self.chat.submitted) == 2)
        self.assertEqual(self.embedder.calls[-2:], [16000, 24000])


if __name__ == "__main__":
    unittest.main()
