"""Push-to-talk and hands-free voice (F022): hold a key to talk, double tap to keep talking
prompt after prompt, tap or "stop listening" to end. Fakes stand in for audio and models."""

import queue
import threading
import time
import unittest
from typing import Any, cast

import numpy as np

from tests.test_voice import SPEECH, WAKE, FakeChat, FakeEngine, FakeMic, FakeSpeaker, FakeTranscriber, wait_for
from voice import build_key
from voice.assistant import TRANSITIONS, VoiceAssistant, VoiceState
from voice.config import VoiceConfig
from voice.dictation import dictated_prompt, hands_free_command
from voice.vad import FRAME_SAMPLES, CommandRecorder
from voice.wakeword import WakeWordDetector

S = VoiceState
QUIET = np.zeros(FRAME_SAMPLES, dtype=np.int16)
LOUD = np.full(FRAME_SAMPLES, 9, dtype=np.int16)


class FakeKey:
    def __init__(self):
        self.events: queue.Queue[str] = queue.Queue()
        self.held = False
        self.on_event = None
        self.started = 0
        self.closed = 0

    def press(self, *events):
        for event in events:
            self.events.put(event)

    @property
    def pending(self):
        return not self.events.empty()

    def poll(self):
        try:
            return self.events.get_nowait()
        except queue.Empty:
            return None

    def clear(self):
        while self.poll() is not None:
            pass

    def start(self):
        self.started += 1

    def close(self):
        self.closed += 1


class FakeRecorder:
    """`audios` feed VAD recordings (wake word and hands-free); `held` feed key-held recordings.
    When the script runs out, a recording waits briefly and finds no speech."""

    def __init__(self, audios=(), held=()):
        self.audios = list(audios)
        self.held = list(held)
        self.held_calls = 0

    def record(self, mic, preroll=None, listen_timeout=10, max_duration=60, give_up=None):
        if give_up is not None and give_up():
            return None

        if self.audios:
            return self.audios.pop(0)

        time.sleep(0.005)
        return None

    def record_held(self, mic, held, preroll=None, max_duration=60):
        self.held_calls += 1
        return self.held.pop(0) if self.held else SPEECH


class ReplyingChat(FakeChat):
    """Answers every prompt at once, as a finished turn would."""

    def submit(self, line):
        result = super().submit(line)

        if self.on_reply is not None and line not in ("y", "n", "stop"):
            self.on_reply(f"Answer to {line}")

        return result


class PushToTalkCase(unittest.TestCase):
    def build(self, texts=(), audios=(), held=(), trigger="push_to_talk", chat=None, frames=()):
        self.now = 0.0
        self.mic = FakeMic(frames)
        self.key = FakeKey()
        self.recorder = FakeRecorder(audios, held)
        self.transcriber = FakeTranscriber(texts)
        self.speaker = FakeSpeaker(self.mic)
        self.chat = chat or FakeChat()
        self.shown = []
        config = VoiceConfig(wake_word="jarvis", voice_trigger=trigger, hands_free_idle_timeout=60)
        detector = (
            WakeWordDetector(FakeEngine(), threshold=0.5, min_frames=1, cooldown=0) if config.uses_wake_word else None
        )
        self.assistant = VoiceAssistant(
            cast(Any, self.chat), self.mic, detector, self.recorder, self.transcriber, self.speaker, config,
            show=self.shown.append, sleep=lambda s: None, playback_grace=0, key=self.key,
            clock=lambda: self.now,
        )  # fmt: skip
        return self.assistant

    def start(self):
        self.thread = threading.Thread(target=self.assistant.run, daemon=True)
        self.thread.start()

    def wait(self, condition, timeout=5.0):
        self.assertTrue(wait_for(condition, timeout), f"condition not reached; shown={self.shown}")

    def tearDown(self):
        if getattr(self, "thread", None) is not None:
            self.assistant.request_stop()
            self.thread.join(5)
            self.assertFalse(self.thread.is_alive(), "assistant did not shut down")

    @property
    def text(self):
        return "\n".join(self.shown)


class HoldToTalkTests(PushToTalkCase):
    def test_held_key_records_and_sends_the_cleaned_prompt(self):
        self.build(texts=["um, open the the readme"])
        self.key.press("down", "hold")
        self.start()
        self.wait(lambda: self.chat.submitted)
        self.assertEqual(self.chat.submitted, ["Open the readme."])
        self.assertEqual(self.key.started, 1)

    def test_a_quick_tap_sends_nothing(self):
        self.build(texts=["should not be used"])
        self.key.press("down", "tap")
        self.start()
        self.wait(lambda: self.recorder.held_calls == 1 and self.assistant.state is S.WAITING_FOR_WAKE_WORD)
        time.sleep(0.05)
        self.assertEqual(self.chat.submitted, [])
        self.assertEqual(self.transcriber.texts, ["should not be used"])

    def test_spoken_stop_and_exit_reach_the_chat_as_commands(self):
        self.build(texts=["Stop."])
        self.key.press("down", "hold")
        self.start()
        self.wait(lambda: self.chat.submitted)
        self.assertEqual(self.chat.submitted, ["stop"])

    def test_spoken_exit_stops_voice_mode(self):
        self.build(texts=["Exit."])
        self.chat.quit = True
        self.key.press("down", "hold")
        self.start()
        self.thread.join(5)
        self.assertEqual(self.chat.submitted, ["exit"])
        self.assertIs(self.assistant.state, S.STOPPING)

    def test_works_without_a_wake_word_model(self):
        self.build(texts=["list the files"])
        self.assertIsNone(self.assistant.detector)
        self.key.press("down", "hold")
        self.start()
        self.wait(lambda: self.chat.submitted)
        self.assertIn("Right Ctrl", self.text)

    def test_wake_word_still_works_next_to_the_key(self):
        self.build(texts=["Jarvis, explain the architecture"], audios=[SPEECH], trigger="both", frames=[WAKE])
        self.start()
        self.wait(lambda: self.chat.submitted)
        self.assertEqual(self.chat.submitted, ["explain the architecture"])  # wake-word prompts are not reworded

    def test_key_press_cuts_off_speech(self):
        self.build()
        self.assistant.state = S.SPEAKING
        assert self.key.on_event is not None
        self.key.on_event("down")
        self.assertEqual(self.speaker.stops, 1)

    def test_key_press_outside_speech_leaves_queued_speech_alone(self):
        self.build()
        self.assistant.state = S.PROCESSING
        assert self.key.on_event is not None
        self.key.on_event("down")
        self.assertEqual(self.speaker.stops, 0)

    def test_new_transitions_exist(self):
        self.assertIn(S.LISTENING_FOR_COMMAND, TRANSITIONS[S.WAITING_FOR_WAKE_WORD])
        self.assertIn(S.LISTENING_FOR_COMMAND, TRANSITIONS[S.PROCESSING])
        self.assertIn(S.SPEAKING, TRANSITIONS[S.LISTENING_FOR_COMMAND])
        self.assertIn(S.LISTENING_FOR_COMMAND, TRANSITIONS[S.RETURN_TO_STANDBY])


class HandsFreeTests(PushToTalkCase):
    def start_hands_free(self):
        self.key.press("down", "tap", "down", "double_tap")
        self.start()
        # The message, not the flag: a short script can turn hands-free off again before a check.
        self.wait(lambda: "[HANDS-FREE] On" in self.text)

    def test_double_tap_takes_prompt_after_prompt(self):
        self.build(texts=["open the readme", "uh, run the tests"], audios=[SPEECH, SPEECH])
        self.start_hands_free()
        self.wait(lambda: len(self.chat.submitted) == 2)
        self.assertEqual(self.chat.submitted, ["Open the readme.", "Run the tests."])
        self.assertIn("[HANDS-FREE] On", self.text)
        self.assertTrue(self.assistant.hands_free)
        self.assertEqual(self.recorder.held_calls, 2)  # the two taps; the prompts needed no key

    def test_answers_are_spoken_between_prompts(self):
        self.build(texts=["open the readme", "run the tests"], audios=[SPEECH, SPEECH], chat=ReplyingChat())
        self.start_hands_free()
        self.wait(lambda: len(self.speaker.said) == 2)
        self.assertEqual(self.speaker.said, ["Answer to Open the readme.", "Answer to Run the tests."])
        self.assertTrue(self.assistant.hands_free)

    def test_saying_stop_listening_ends_hands_free(self):
        self.build(texts=["open the readme", "stop listening"], audios=[SPEECH, SPEECH])
        self.start_hands_free()
        self.wait(lambda: not self.assistant.hands_free)
        self.assertEqual(self.chat.submitted, ["Open the readme."])
        self.assertIn("[HANDS-FREE] Off", self.text)

    def test_a_tap_ends_hands_free(self):
        self.build()
        self.start_hands_free()
        self.key.press("down", "tap")
        self.wait(lambda: not self.assistant.hands_free)
        self.wait(lambda: self.assistant.state is S.WAITING_FOR_WAKE_WORD)

    def test_holding_the_key_in_hands_free_sends_one_prompt_and_stays_on(self):
        self.build(texts=["show the diff"])
        self.start_hands_free()
        self.key.press("down", "hold")
        self.wait(lambda: self.chat.submitted)
        self.assertEqual(self.chat.submitted, ["Show the diff."])
        self.assertTrue(self.assistant.hands_free)

    def test_long_silence_ends_hands_free(self):
        self.build()
        self.start_hands_free()
        self.now += 61
        self.wait(lambda: not self.assistant.hands_free)
        self.assertIn("no speech", self.text)

    def test_noise_is_ignored_and_listening_goes_on(self):
        self.build(texts=["um", "open the readme"], audios=[SPEECH, SPEECH])
        self.start_hands_free()
        self.wait(lambda: self.chat.submitted)
        self.assertEqual(self.chat.submitted, ["Open the readme."])
        self.assertTrue(self.assistant.hands_free)

    def test_wake_word_command_turns_hands_free_on(self):
        self.build(texts=["Jarvis, hands free"], audios=[SPEECH], trigger="both", frames=[WAKE])
        self.start()
        self.wait(lambda: self.assistant.hands_free)
        self.assertEqual(self.chat.submitted, [])

    def test_approval_is_asked_and_answered_in_hands_free(self):
        chat: Any = FakeChat()
        request = {"calls": [{"name": "terminal", "args": {"command": "ls"}}]}

        def submit(line):
            chat.submitted.append(line)
            chat.pending_approval = request if line == "List the files." else None
            return True

        chat.submit = submit
        self.build(texts=["list the files", "yes"], audios=[SPEECH, SPEECH], chat=chat)
        self.start_hands_free()
        self.wait(lambda: len(chat.submitted) == 2)
        self.assertEqual(chat.submitted, ["List the files.", "y"])
        self.assertIn("terminal", self.speaker.said[0])
        self.assertTrue(self.assistant.hands_free)


class DictationCommandTests(unittest.TestCase):
    def test_hands_free_commands(self):
        for text in ("hands free", "Hands-free mode.", "turn on hands free", "chế độ rảnh tay", "rảnh tay"):
            with self.subTest(text=text):
                self.assertEqual(hands_free_command(text), "on")

        for text in ("Stop listening.", "hands free off", "dừng nghe", "tắt rảnh tay"):
            with self.subTest(text=text):
                self.assertEqual(hands_free_command(text), "off")

        self.assertIsNone(hands_free_command("is hands free mode safe to use"))

    def test_dictated_prompt_keeps_commands_plain(self):
        self.assertEqual(dictated_prompt("Stop."), "stop")
        self.assertEqual(dictated_prompt("cancel that"), "stop")
        self.assertEqual(dictated_prompt("Quit."), "quit")
        self.assertEqual(dictated_prompt("um open it"), "Open it.")


class FakeVAD:
    def is_speech(self, frame):
        return bool(frame[0])

    def reset(self):
        pass


class RecorderTests(unittest.TestCase):
    def test_record_held_keeps_everything_until_release(self):
        mic = FakeMic([LOUD, QUIET, QUIET, QUIET, LOUD])
        held = iter([True] * 5 + [False])
        audio = CommandRecorder(FakeVAD()).record_held(mic, lambda: next(held), preroll=[QUIET])
        assert audio is not None
        self.assertEqual(len(audio), 6 * FRAME_SAMPLES)  # silence does not end it

    def test_record_held_with_nothing_heard(self):
        self.assertIsNone(CommandRecorder(FakeVAD()).record_held(FakeMic([]), lambda: False))

    def test_record_held_stops_at_the_limit(self):
        mic = FakeMic([LOUD] * 100)
        audio = CommandRecorder(FakeVAD()).record_held(mic, lambda: True, max_duration=0.8)
        assert audio is not None
        self.assertEqual(len(audio), 10 * FRAME_SAMPLES)

    def test_give_up_before_speech_returns_none(self):
        mic = FakeMic([QUIET] * 50)
        calls = []
        audio = CommandRecorder(FakeVAD()).record(mic, give_up=lambda: calls.append(1) or len(calls) > 3)
        self.assertIsNone(audio)

    def test_give_up_is_ignored_once_speech_started(self):
        mic = FakeMic([LOUD, LOUD] + [QUIET] * 20)
        checks = iter([False])  # then True: wants to give up from the second frame on
        audio = CommandRecorder(FakeVAD(), end_silence=0.4).record(mic, give_up=lambda: next(checks, True))
        self.assertIsNotNone(audio)


class BuildKeyTests(unittest.TestCase):
    def test_no_key_when_push_to_talk_is_off(self):
        self.assertIsNone(build_key(VoiceConfig(voice_trigger="wake_word"), print))

    def test_key_watcher_for_the_configured_key(self):
        names = []
        watcher = build_key(VoiceConfig(push_to_talk_key="f9"), print, reader=lambda name: names.append(name) or (lambda: False))
        self.assertIsNotNone(watcher)
        self.assertEqual(names, ["f9"])

    def test_unavailable_key_falls_back_to_the_wake_word(self):
        def broken(name):
            raise OSError("not Windows")

        shown = []
        self.assertIsNone(build_key(VoiceConfig(voice_trigger="both"), shown.append, reader=broken))
        self.assertIn("wake word only", shown[0])

    def test_unavailable_key_without_a_wake_word_is_an_error(self):
        def broken(name):
            raise OSError("not Windows")

        with self.assertRaises(RuntimeError):
            build_key(VoiceConfig(voice_trigger="push_to_talk"), print, reader=broken)


if __name__ == "__main__":
    unittest.main()
