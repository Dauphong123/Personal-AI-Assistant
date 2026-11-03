"""Voice mode tests. No microphone, speaker, or speech model is needed: every audio
component is a fake, and the real ChatLoop + graph are used where the agent matters."""

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from tests import stubs

stubs.install()

from chat_loop import ChatLoop  # noqa: E402
from graph import ModelGraph, node  # noqa: E402
from jobs import JobManager  # noqa: E402
from voice.assistant import (  # noqa: E402
    InvalidTransition,
    VoiceAssistant,
    VoiceState,
    describe_approval,
)
from voice.audio import MicDisconnected  # noqa: E402
from voice.config import VoiceConfig, VoiceConfigError  # noqa: E402
from voice.wakeword import WakeWordDetector  # noqa: E402

S = VoiceState
WAKE = np.ones(1280, dtype=np.int16)
QUIET = np.zeros(1280, dtype=np.int16)
ECHO = WAKE.copy()  # the assistant's own voice, as heard by the microphone
SPEECH = np.full(8000, 5, dtype=np.int16)


def wait_for(condition, timeout=5.0):
    end = time.monotonic() + timeout

    while time.monotonic() < end:
        if condition():
            return True

        time.sleep(0.005)

    return False


class FakeMic:
    def __init__(self, frames=()):
        self.frames = list(frames)  # np frames, None, or exceptions to raise
        self.start_calls = 0
        self.start_failures = 0
        self.closed = 0
        self.flushes = 0

    def start(self):
        self.start_calls += 1

        if self.start_failures:
            self.start_failures -= 1
            raise OSError("no device")

    def read(self, timeout=0.1):
        if self.frames:
            item = self.frames.pop(0)

            if isinstance(item, Exception):
                raise item

            return item

        time.sleep(0.002)
        return None

    def flush(self):
        self.flushes += 1
        # Scripted frames stand for audio that has not been spoken yet; only echo is queued audio.
        self.frames = [f for f in self.frames if f is not ECHO]

    def close(self):
        self.closed += 1


class FakeEngine:
    def __init__(self):
        self.resets = 0

    def score(self, frame):
        return 0.9 if frame[0] == 1 else 0.0

    def reset(self):
        self.resets += 1


class FakeRecorder:
    def __init__(self, audios):
        self.audios = list(audios)
        self.calls = []

    def record(self, mic, preroll=None, listen_timeout=10, max_duration=60):
        self.calls.append(SimpleNamespace(preroll=preroll))
        return self.audios.pop(0) if self.audios else None


class FakeTranscriber:
    def __init__(self, texts):
        self.texts = list(texts)

    def transcribe(self, audio):
        item = self.texts.pop(0)

        if isinstance(item, Exception):
            raise item

        return item


class FakeSpeaker:
    def __init__(self, mic=None):
        self.said = []
        self.beeps = 0
        self.stops = 0
        self.closed = 0
        self.fail = False
        self.mic = mic
        self.echo = False

    def say(self, text):
        self.said.append(text)

        if self.echo and self.mic is not None:
            self.mic.frames.append(ECHO)  # the assistant's own voice reaching the microphone

        if self.fail:
            raise RuntimeError("no audio device")

    def beep(self):
        self.beeps += 1

    def stop(self):
        self.stops += 1

    def close(self):
        self.closed += 1


class FakeChat:
    def __init__(self):
        self.submitted = []
        self.busy = False
        self.pending_approval = None
        self.on_reply = None
        self.quit = False

    def submit(self, line):
        self.submitted.append(line)
        return not self.quit


class VoiceTestCase(unittest.TestCase):
    def build(self, frames=(), audios=(), texts=(), chat=None, wake_word="jarvis", min_frames=1, verifier=None, language="en"):
        self.mic = FakeMic(frames)
        self.engine = FakeEngine()
        self.detector = WakeWordDetector(self.engine, threshold=0.5, min_frames=min_frames, cooldown=0)
        self.recorder = FakeRecorder(audios)
        self.transcriber = FakeTranscriber(texts)
        self.speaker = FakeSpeaker(self.mic)
        self.chat = chat or FakeChat()
        self.shown = []
        config = VoiceConfig(wake_word=wake_word, stt_language=language)
        self.assistant = VoiceAssistant(
            self.chat, self.mic, self.detector, self.recorder, self.transcriber,
            self.speaker, config, show=self.shown.append, sleep=lambda s: None, playback_grace=0,
            verifier=verifier,
        )  # fmt: skip
        return self.assistant

    def run_until(self, condition, timeout=5.0):
        thread = threading.Thread(target=self.assistant.run, daemon=True)
        thread.start()
        reached = wait_for(condition, timeout)
        self.assistant.request_stop()
        thread.join(5)
        self.assertFalse(thread.is_alive(), "assistant did not shut down")
        self.assertTrue(reached, f"condition not reached; shown={self.shown}")

    @property
    def text(self):
        return "\n".join(self.shown)


class WakeWordFlowTests(VoiceTestCase):
    def test_wake_word_then_command_is_forwarded(self):
        self.build(frames=[WAKE], audios=[SPEECH], texts=["Jarvis, explain the architecture"])
        self.run_until(lambda: self.chat.submitted)
        self.assertEqual(self.chat.submitted, ["explain the architecture"])
        self.assertIn("[ACTIVATED] Wake word detected.", self.text)
        self.assertIn("You said: explain the architecture", self.text)
        self.assertEqual(self.speaker.beeps, 0, "an acknowledgment would talk over the command")

    def test_speech_after_the_wake_word_is_not_discarded(self):
        self.build(frames=[QUIET, QUIET, WAKE], audios=[SPEECH], texts=["Jarvis, hello there"])
        self.run_until(lambda: self.chat.submitted)
        self.assertTrue(self.recorder.calls[0].preroll, "recorder must receive the frames around the wake word")

    def test_no_wake_word_means_nothing_happens(self):
        self.build(frames=[QUIET] * 50)
        self.run_until(lambda: not self.mic.frames)
        self.assertEqual(self.recorder.calls, [])
        self.assertEqual(self.chat.submitted, [])
        self.assertIs(self.assistant.state, S.STOPPING)
        self.assertNotIn("ACTIVATED", self.text)

    def test_false_detection_without_speech_returns_to_standby(self):
        self.build(frames=[WAKE], audios=[None])
        self.run_until(lambda: self.assistant.cycles >= 1)
        self.assertEqual(self.chat.submitted, [])
        self.assertIn("No speech heard", self.text)
        self.assertEqual(self.text.count("[STANDBY] Waiting for Jarvis"), 2)

    def test_delayed_command_after_wake_word_alone(self):
        self.build(frames=[WAKE], audios=[SPEECH, SPEECH], texts=["Jarvis.", "what time is it"])
        self.run_until(lambda: self.chat.submitted)
        self.assertEqual(self.chat.submitted, ["what time is it"])
        self.assertEqual(self.speaker.beeps, 1)

    def test_recognizer_noise_after_the_wake_word_is_not_sent_to_the_agent(self):
        self.build(frames=[WAKE], audios=[SPEECH, SPEECH], texts=["you", "what time is it"])
        self.run_until(lambda: self.chat.submitted)
        self.assertEqual(self.chat.submitted, ["what time is it"])
        self.assertEqual(self.speaker.beeps, 1)

    def test_empty_speech_is_rejected(self):
        self.build(frames=[WAKE], audios=[SPEECH, SPEECH], texts=["", ""])
        self.run_until(lambda: self.assistant.cycles >= 1)
        self.assertEqual(self.chat.submitted, [])
        self.assertIn("Nothing recognized", self.text)

    def test_failed_transcription_recovers_and_next_cycle_works(self):
        self.build(
            frames=[WAKE, WAKE],
            audios=[SPEECH, SPEECH],
            texts=[RuntimeError("model crashed"), "Jarvis, try again"],
        )
        self.run_until(lambda: self.chat.submitted)
        self.assertIn("[ERROR] Speech recognition failed.", self.text)
        self.assertEqual(self.chat.submitted, ["try again"])

    def test_repeated_cycles(self):
        self.build(
            frames=[WAKE, WAKE, WAKE],
            audios=[SPEECH] * 3,
            texts=["Jarvis one", "Jarvis two", "Jarvis three"],
        )
        self.run_until(lambda: len(self.chat.submitted) == 3)
        self.assertEqual(self.chat.submitted, ["one", "two", "three"])

    def test_spoken_quit_stops_the_assistant(self):
        chat = FakeChat()
        chat.quit = True
        self.build(frames=[WAKE], audios=[SPEECH], texts=["Jarvis, quit"], chat=chat)
        thread = threading.Thread(target=self.assistant.run, daemon=True)
        thread.start()
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertIs(self.assistant.state, S.STOPPING)


class SpeakingTests(VoiceTestCase):
    def test_reply_is_spoken_then_returns_to_standby(self):
        self.build()
        self.chat.busy = True
        self.chat.pending_approval = None
        thread = threading.Thread(target=self.assistant.run, daemon=True)
        thread.start()
        self.chat.on_reply("The answer is 42.")
        time.sleep(0.05)
        self.chat.busy = False
        self.assertTrue(wait_for(lambda: self.assistant.state is S.WAITING_FOR_WAKE_WORD and self.assistant.cycles >= 1))
        self.assistant.request_stop()
        thread.join(5)
        self.assertEqual(self.speaker.said, ["The answer is 42."])
        self.assertIn("[SPEAKING] Playing response...", self.text)

    def test_assistant_does_not_activate_from_its_own_voice(self):
        self.build()
        self.speaker.echo = True  # every utterance leaks a wake-word frame into the mic
        thread = threading.Thread(target=self.assistant.run, daemon=True)
        thread.start()
        self.chat.on_reply("Hello. I am here.")
        self.assertTrue(wait_for(lambda: self.speaker.said and self.assistant.cycles >= 1))
        time.sleep(0.1)
        self.assistant.request_stop()
        thread.join(5)
        self.assertEqual(self.recorder.calls, [], "the echoed wake word must be flushed, not detected")
        self.assertNotIn("ACTIVATED", self.text)

    def test_playback_failure_does_not_crash(self):
        self.build()
        self.speaker.fail = True
        thread = threading.Thread(target=self.assistant.run, daemon=True)
        thread.start()
        self.chat.on_reply("Hi there.")
        self.assertTrue(wait_for(lambda: self.assistant.cycles >= 1))
        self.assistant.request_stop()
        thread.join(5)
        self.assertIn("Could not play the response", self.text)


class ApprovalTests(VoiceTestCase):
    def setUp(self):
        self.request = {"calls": [{"name": "write_file", "args": {"path": "a.txt", "content": "x"}}]}

    def test_description_is_speakable(self):
        text = describe_approval(self.request)
        self.assertIn("write file", text)
        self.assertIn("Say yes", text)

    def run_approval(self, answer_text, audios=(SPEECH,)):
        chat = FakeChat()
        chat.busy = True
        chat.pending_approval = self.request
        self.build(audios=audios, texts=[answer_text] if answer_text is not None else [], chat=chat)
        self.run_until(lambda: chat.submitted)

    def test_spoken_yes_approves(self):
        self.run_approval("Yes.")
        self.assertEqual(self.chat.submitted, ["y"])
        self.assertIn("Say yes", self.speaker.said[0])

    def test_unclear_answer_declines(self):
        self.run_approval("hmm let me think")
        self.assertEqual(self.chat.submitted, ["n"])

    def test_silence_declines(self):
        self.run_approval(None, audios=[None])
        self.assertEqual(self.chat.submitted, ["n"])

    def test_spoken_stop_while_waiting_is_forwarded_as_stop(self):
        self.run_approval("stop")
        self.assertEqual(self.chat.submitted, ["stop"])


class ProcessingTests(VoiceTestCase):
    def test_wake_word_while_agent_is_busy_can_stop_it(self):
        chat = FakeChat()
        chat.busy = True
        self.build(frames=[WAKE], audios=[SPEECH], texts=["Jarvis, stop"], chat=chat)
        self.run_until(lambda: chat.submitted)
        self.assertEqual(chat.submitted, ["stop"])


class StateMachineTests(VoiceTestCase):
    def test_illegal_transition_is_rejected(self):
        self.build()
        with self.assertRaises(InvalidTransition):
            self.assistant._transition(S.SPEAKING)

    def test_state_progression_is_visible_to_the_user(self):
        self.build(frames=[WAKE], audios=[SPEECH], texts=["Jarvis, hi"])
        self.run_until(lambda: self.chat.submitted)
        order = [
            "[STANDBY] Waiting for Jarvis...",
            "[ACTIVATED]",
            "[LISTENING]",
            "[TRANSCRIBING]",
            "[PROCESSING]",
        ]
        positions = [self.text.index(m) for m in order]
        self.assertEqual(positions, sorted(positions))


class RobustnessTests(VoiceTestCase):
    def test_microphone_disconnection_recovers(self):
        self.build(frames=[MicDisconnected("unplugged"), WAKE], audios=[SPEECH], texts=["Jarvis, still here"])
        self.run_until(lambda: self.chat.submitted)
        self.assertEqual(self.mic.start_calls, 2)
        self.assertIn("disconnected", self.text)
        self.assertEqual(self.chat.submitted, ["still here"])

    def test_missing_microphone_stops_cleanly_after_retries(self):
        self.build()
        self.mic.start_failures = 99
        self.assistant.run()
        self.assertIs(self.assistant.state, S.STOPPING)
        self.assertIn("Could not open the microphone", self.text)
        self.assertEqual(self.mic.closed, 1)

    def test_unrecoverable_disconnection_stops(self):
        self.build(frames=[MicDisconnected("gone")])
        self.mic.start_failures = 0
        original = self.mic.start

        def start():
            if self.mic.start_calls >= 1:
                self.mic.start_calls += 1
                raise OSError("still gone")

            original()

        self.mic.start = start
        self.assistant.run()
        self.assertIs(self.assistant.state, S.STOPPING)

    def test_unexpected_error_returns_to_standby_and_continues(self):
        self.build(frames=[WAKE, WAKE], audios=[SPEECH, SPEECH], texts=[ValueError("boom"), "Jarvis, ok"])
        with patch.object(self.transcriber, "transcribe", side_effect=[ValueError("x"), "Jarvis, ok"]):
            self.run_until(lambda: self.chat.submitted)

        self.assertEqual(self.chat.submitted, ["ok"])

    def test_clean_shutdown_releases_audio_and_stops_playback(self):
        self.build()
        thread = threading.Thread(target=self.assistant.run, daemon=True)
        thread.start()
        wait_for(lambda: self.mic.start_calls == 1)
        self.assistant.request_stop()
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.mic.closed, 1)
        self.assertEqual(self.speaker.closed, 1)
        self.assertIs(self.assistant.state, S.STOPPING)


class RealAgentTests(VoiceTestCase):
    """Voice mode driving the real ChatLoop and LangGraph graph (model chains stubbed)."""

    def setUp(self):
        self.seen = []
        self.replies = []

        def direct_chain(inputs):
            self.seen.append([m.content for m in inputs["messages"]])
            reply = self.replies.pop(0)

            if isinstance(reply, Exception):
                raise reply

            return reply

        router = RunnableLambda(lambda x: SimpleNamespace(destination="direct"))
        for p in [
            patch.object(node, "router_chain", router),
            patch.object(node, "direct_chain", RunnableLambda(direct_chain)),
        ]:
            p.start()
            self.addCleanup(p.stop)

        self.app = ModelGraph.compile(checkpointer=InMemorySaver())
        self.out = []
        self.chat = ChatLoop(self.app, {"configurable": {"thread_id": "voice"}}, JobManager(), write=self.out.append)
        self.chat.start()
        self.addCleanup(self.chat.close)

    def test_transcript_reaches_the_graph_and_reply_is_spoken(self):
        self.replies = [AIMessage(content="Your project has three layers.")]
        self.build(frames=[WAKE], audios=[SPEECH], texts=["Jarvis, explain my project"], chat=self.chat)
        self.run_until(lambda: self.speaker.said)
        self.assertEqual(self.seen[0], ["explain my project"])
        self.assertEqual(self.speaker.said, ["Your project has three layers."])

    def test_conversation_state_is_kept_across_voice_turns_and_keyboard(self):
        self.replies = [AIMessage(content="one"), AIMessage(content="two"), AIMessage(content="three")]
        self.build(
            frames=[WAKE, WAKE], audios=[SPEECH, SPEECH],
            texts=["Jarvis, first question", "Jarvis, second question"], chat=self.chat,
        )  # fmt: skip
        self.run_until(lambda: len(self.speaker.said) == 2)
        self.assertEqual(self.seen[1], ["first question", "one", "second question"])

        # The keyboard keeps working and shares the same conversation.
        self.chat.submit("typed question")
        self.assertTrue(wait_for(lambda: len(self.seen) == 3))
        self.assertEqual(
            self.seen[2],
            ["first question", "one", "second question", "two", "typed question"],
        )

    def test_agent_failure_is_reported_and_assistant_recovers(self):
        self.replies = [RuntimeError("model down"), AIMessage(content="back again")]
        self.build(
            frames=[WAKE, WAKE], audios=[SPEECH, SPEECH],
            texts=["Jarvis, break", "Jarvis, retry"], chat=self.chat,
        )  # fmt: skip
        self.run_until(lambda: len(self.speaker.said) == 2)
        self.assertIn("Sorry", self.speaker.said[0])
        self.assertEqual(self.speaker.said[1], "back again")


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        config = VoiceConfig.from_env({})
        self.assertFalse(config.enabled)
        self.assertEqual(config.wake_word, "hey_jarvis")
        self.assertIsNone(config.microphone_device)

    def test_values_are_read_and_typed(self):
        config = VoiceConfig.from_env(
            {
                "VOICE_ENABLED": "true",
                "WAKE_WORD_THRESHOLD": "0.7",
                "MICROPHONE_DEVICE": "2",
                "AUDIO_OUTPUT_DEVICE": "Speakers",
                "SPEECH_LANGUAGE": "VI",
            }
        )
        self.assertTrue(config.enabled)
        self.assertEqual(config.wake_word_threshold, 0.7)
        self.assertEqual(config.microphone_device, 2)
        self.assertEqual(config.output_device, "Speakers")
        self.assertEqual(config.stt_language, "vi")

    def test_invalid_values_are_rejected(self):
        for env in [
            {"WAKE_WORD_THRESHOLD": "2"},
            {"WAKE_WORD_THRESHOLD": "high"},
            {"VOICE_ENABLED": "maybe"},
            {"VOICE_LISTENING_TIMEOUT": "0"},
            {"WAKE_WORD": "missing_model.onnx"},
        ]:
            with self.assertRaises(VoiceConfigError, msg=str(env)):
                VoiceConfig.from_env(env)


if __name__ == "__main__":
    unittest.main()
