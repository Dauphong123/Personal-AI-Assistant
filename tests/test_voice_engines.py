"""Real-model integration test, skipped unless RUN_VOICE_ENGINE_TESTS=1.

Piper synthesizes "Hey Jarvis, ..." and that audio is fed to the real wake-word model,
VAD, recorder and Whisper through the real VoiceAssistant, ChatLoop and graph. Only the
microphone and the loudspeaker are fake. The first run downloads about 200 MB of models.
"""

import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tests import stubs

stubs.install()

try:
    import faster_whisper  # noqa: F401
    import openwakeword  # noqa: F401
    import piper  # noqa: F401
    import silero_vad  # noqa: F401
    from scipy.signal import resample_poly

    AVAILABLE = os.environ.get("RUN_VOICE_ENGINE_TESTS") == "1"
except ImportError:
    AVAILABLE = False

if AVAILABLE:
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda
    from langgraph.checkpoint.memory import InMemorySaver

    from chat_loop import ChatLoop
    from graph import ModelGraph, node
    from jobs import JobManager
    from tests.test_voice import FakeMic, wait_for
    from voice.assistant import VoiceAssistant
    from voice.config import VoiceConfig
    from voice.speaker import SpeakerEmbedder, SpeakerVerifier, VoicePrint
    from voice.stt import FasterWhisperTranscriber
    from voice.tts import PiperSynthesizer, Speaker
    from voice.vad import CommandRecorder, SileroVAD
    from voice.wakeword import FRAME_SAMPLES, OpenWakeWordEngine, WakeWordDetector


class RecordingPlayer:
    def __init__(self):
        self.clips = []

    def play(self, samples, rate, should_stop=lambda: False):
        self.clips.append(len(samples) / rate)

    def close(self):
        pass


@unittest.skipUnless(AVAILABLE, "set RUN_VOICE_ENGINE_TESTS=1 (needs requirements_voice.txt)")
class PipelineCase(unittest.TestCase):
    """Shared helpers: fake microphone frames in, the real voice pipeline and graph, replies out."""

    reply = "It is a quarter past three."

    @classmethod
    def make_config(cls):
        return VoiceConfig(wake_word="hey_jarvis")

    @classmethod
    def setUpClass(cls):
        cls.config = cls.make_config()
        cls.synth = PiperSynthesizer(cls.config)
        cls.engine = OpenWakeWordEngine(cls.config.wake_word)
        cls.vad = SileroVAD(cls.config.vad_threshold)
        cls.stt = FasterWhisperTranscriber(cls.config)

    def speech(self, text):
        audio, rate = self.synth.synthesize(text)
        return resample_poly(audio.astype(np.float32), 16000, rate).astype(np.int16)

    def frames(self, *parts):
        stream = np.concatenate(parts)
        usable = len(stream) - len(stream) % FRAME_SAMPLES
        return [stream[i : i + FRAME_SAMPLES] for i in range(0, usable, FRAME_SAMPLES)]

    def make_speaker(self):
        return Speaker(self.synth, self.player)

    def run_voice(self, frames, expected_replies=1, timeout=120, verifier=None):
        self.seen = []
        self.engine.reset()

        def direct_chain(inputs):
            self.seen.append(inputs["messages"][-1].content)
            return AIMessage(content=self.reply)

        router = RunnableLambda(lambda x: SimpleNamespace(destination="direct"))

        with (
            patch.object(node, "router_chain", router),
            patch.object(node, "direct_chain", RunnableLambda(direct_chain)),
        ):
            app = ModelGraph.compile(checkpointer=InMemorySaver())
            chat = ChatLoop(app, {"configurable": {"thread_id": "e2e"}}, JobManager(), write=lambda s: None)
            chat.start()
            self.addCleanup(chat.close)

            self.player = RecordingPlayer()
            mic = FakeMic(frames)
            assistant = VoiceAssistant(
                chat,
                mic,
                WakeWordDetector(self.engine, self.config.wake_word_threshold, 2, 2.0),
                CommandRecorder(self.vad, self.config.end_silence),
                self.stt,
                self.make_speaker(),
                self.config,
                show=lambda s: None,
                sleep=lambda s: None,
                playback_grace=0,
                verifier=verifier,
            )
            thread = threading.Thread(target=assistant.run, daemon=True)
            thread.start()
            # The "go ahead" beep (0.12 s) is also played through the player; only a spoken reply counts.
            replies = lambda: sum(seconds > 0.3 for seconds in self.player.clips)  # noqa: E731
            done = wait_for(lambda: replies() >= expected_replies, timeout=timeout)
            assistant.request_stop()
            thread.join(10)
            self.assertFalse(thread.is_alive())
            return done

@unittest.skipUnless(AVAILABLE, "set RUN_VOICE_ENGINE_TESTS=1 (needs requirements_voice.txt)")
class RealEngineTests(PipelineCase):
    def test_wake_word_and_command_in_one_breath(self):
        silence = np.zeros(16000, dtype=np.int16)
        frames = self.frames(silence, self.speech("Hey Jarvis, what time is it?"), silence, silence)
        self.assertTrue(self.run_voice(frames))
        self.assertEqual(len(self.seen), 1)
        self.assertIn("time", self.seen[0].lower())
        self.assertNotIn("jarvis", self.seen[0].lower())

    def test_wake_word_then_command_after_a_pause(self):
        silence = np.zeros(16000, dtype=np.int16)
        pause = np.zeros(24000, dtype=np.int16)
        frames = self.frames(
            silence, self.speech("Hey Jarvis."), pause,
            self.speech("What time is it?"), silence, silence,
        )  # fmt: skip
        self.assertTrue(self.run_voice(frames))
        self.assertEqual(len(self.seen), 1)
        self.assertIn("time", self.seen[0].lower())

    def test_ordinary_speech_does_not_reach_the_agent(self):
        silence = np.zeros(16000, dtype=np.int16)
        frames = self.frames(silence, self.speech("Please remember to buy milk and eggs tomorrow."), silence)
        self.assertFalse(self.run_voice(frames, timeout=10))
        self.assertEqual(self.seen, [])


@unittest.skipUnless(AVAILABLE, "set RUN_VOICE_ENGINE_TESTS=1 (needs requirements_voice.txt)")
class RealSpeakerVerificationTests(PipelineCase):
    """Enroll one synthetic voice; accept it and reject a different one, end to end."""

    ENROLL = [
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "Please open the project folder and show me the latest changes.",
        "What is the weather going to be like tomorrow morning?",
        "Remind me to review the pull request before lunch today.",
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.stranger = PiperSynthesizer(VoiceConfig(tts_voice="en_US-amy-medium"))
        embedder = SpeakerEmbedder(cls.config.model_dir)
        cls.verifier = SpeakerVerifier(
            embedder, VoicePrint.from_embeddings([embedder.embed(cls.clip(cls.synth, t)) for t in cls.ENROLL])
        )

    @staticmethod
    def clip(synth, text):
        audio, rate = synth.synthesize(text)
        return resample_poly(audio.astype(np.float32), 16000, rate).astype(np.int16)

    def test_scores_separate_the_enrolled_voice_from_another(self):
        said = "Hey Jarvis, what is the weather like today?"
        owner = self.verifier.score(self.clip(self.synth, said))
        other = self.verifier.score(self.clip(self.stranger, said))
        print(f"\n  owner {owner:.2f}  stranger {other:.2f}  threshold {self.verifier.threshold:.2f}")
        self.assertGreater(owner, self.verifier.threshold)
        self.assertLess(other, self.verifier.threshold)

    def pipeline_frames(self, synth):
        silence = np.zeros(16000, dtype=np.int16)
        speech = self.clip(synth, "Hey Jarvis, what time is it?")
        return self.frames(silence, speech, silence, silence)

    def test_enrolled_voice_reaches_the_agent(self):
        self.assertTrue(self.run_voice(self.pipeline_frames(self.synth), verifier=self.verifier))
        self.assertEqual(len(self.seen), 1)

    def test_another_voice_is_rejected_before_the_agent(self):
        self.assertFalse(self.run_voice(self.pipeline_frames(self.stranger), timeout=15, verifier=self.verifier))
        self.assertEqual(self.seen, [])


class CountingSynth:
    """Wraps a real synthesizer and counts how often it was used."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def synthesize(self, text):
        self.calls += 1
        return self.inner.synthesize(text)


@unittest.skipUnless(AVAILABLE, "set RUN_VOICE_ENGINE_TESTS=1 (needs requirements_voice.txt)")
class RealVietnameseTests(PipelineCase):
    """SPEECH_LANGUAGE=vi,en: Vietnamese or English speech in (Whisper small), Vietnamese voice out (Piper)."""

    reply = "Bây giờ là ba giờ mười lăm phút."

    @classmethod
    def make_config(cls):
        return VoiceConfig(wake_word="hey_jarvis", stt_language="vi,en", stt_model="small")

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.vi_synth = PiperSynthesizer(cls.config, cls.config.tts_voice_vi)

    def make_speaker(self):
        self.english, self.vietnamese = CountingSynth(self.synth), CountingSynth(self.vi_synth)
        return Speaker(self.english, self.player, synthesizers={"vi": self.vietnamese})

    def vi_speech(self, text):
        audio, rate = self.vi_synth.synthesize(text)
        return resample_poly(audio.astype(np.float32), 16000, rate).astype(np.int16)

    def command_frames(self, spoken):
        silence = np.zeros(16000, dtype=np.int16)
        gap = np.zeros(4000, dtype=np.int16)
        return self.frames(silence, self.speech("Hey Jarvis."), gap, spoken, silence, silence)

    def test_vietnamese_command_reaches_the_agent_without_the_wake_word(self):
        frames = self.command_frames(self.vi_speech("Hãy chạy các bài kiểm tra."))
        self.assertTrue(self.run_voice(frames))
        print(f"\n  agent received: {self.seen[0]!r}")
        # A real model sometimes mishears a word ("kiểm" as "tiệm"), so most of the words must arrive.
        words = ["hãy", "chạy", "bài", "kiểm", "tra"]
        self.assertGreaterEqual(sum(w in self.seen[0].lower() for w in words), 3, self.seen[0])
        self.assertNotIn("jarvis", self.seen[0].lower())

    def test_vietnamese_reply_uses_the_vietnamese_voice(self):
        frames = self.command_frames(self.vi_speech("Bây giờ là mấy giờ rồi?"))
        self.assertTrue(self.run_voice(frames))
        self.assertGreater(self.vietnamese.calls, 0)
        self.assertEqual(self.english.calls, 0)

    def test_english_still_works_in_vietnamese_mode(self):
        silence = np.zeros(16000, dtype=np.int16)
        frames = self.frames(silence, self.speech("Hey Jarvis, what time is it?"), silence, silence)
        self.assertTrue(self.run_voice(frames))
        self.assertIn("time", self.seen[0].lower())


if __name__ == "__main__":
    unittest.main()
