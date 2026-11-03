"""Regression tests for problems found in the second review of the voice code."""

import io
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from tests import stubs

stubs.install()

from language import detect_language  # noqa: E402
from tests.test_voice import SPEECH, WAKE, FakeChat, FakeMic, VoiceTestCase, wait_for  # noqa: E402
from voice import speaker as speaker_module  # noqa: E402
from voice import text as vtext  # noqa: E402
from voice.assistant import VoiceState as S  # noqa: E402
from voice.audio import AudioPlayer  # noqa: E402

import numpy as np  # noqa: E402


class ApprovalAnswerTests(unittest.TestCase):
    def test_a_second_thought_is_not_an_approval(self):
        for answer in [
            "yes, no wait, don't", "Yes... actually no", "yeah I'm not sure", "yes but only if it is safe",
            "yes wait", "maybe yes", "yep, hold on", "có... không, đừng", "có nhưng khoan đã", "vâng, chưa",
            "okay never mind",
        ]:  # fmt: skip
            self.assertNotEqual(vtext.normalize_answer(answer), "y", answer)

    def test_a_long_ramble_is_not_an_approval(self):
        self.assertEqual(vtext.normalize_answer("yes " * 9), "n")
        self.assertEqual(vtext.normalize_answer("yes please go ahead and do all of that right now"), "n")

    def test_clear_approvals_still_work(self):
        for answer in ["yes", "Yes.", "yeah", "Yes, allow it.", "okay", "go ahead", "sure", "approve", "Có.", "Có, cho phép.", "Đồng ý", "Được rồi", "vâng ạ"]:
            self.assertEqual(vtext.normalize_answer(answer), "y", answer)

    def test_stop_and_refusal_are_unchanged(self):
        self.assertEqual(vtext.normalize_answer("stop"), "stop")
        self.assertEqual(vtext.normalize_answer("dừng lại"), "stop")

        for answer in ["no", "nope", "không", "let me think", ""]:
            self.assertEqual(vtext.normalize_answer(answer), "n", answer)


class TruncationLanguageTests(unittest.TestCase):
    def test_the_suffix_matches_the_language_of_what_was_kept(self):
        english = vtext.clean_for_speech("This is a sentence. " * 50, 100)
        self.assertTrue(english.endswith("The rest is on screen."))

        vietnamese = vtext.clean_for_speech("Đây là một câu tiếng Việt. " * 50, 100)
        self.assertTrue(vietnamese.endswith("Phần còn lại hiển thị trên màn hình."))
        self.assertNotIn("The rest is on screen", vietnamese)


class LanguageDetectionTests(unittest.TestCase):
    def test_a_quoted_vietnamese_name_does_not_make_english_vietnamese(self):
        for text in [
            "The user Đặng Văn said hi in the team chat today",
            "Please ask Nguyễn about the deployment schedule next week",
            "café au lait",
            "Où est le café près de l'école?",
        ]:
            self.assertEqual(detect_language(text), "en", text)

    def test_real_vietnamese_is_still_recognized(self):
        for text in ["Xin chào, bạn có khỏe không?", "Bây giờ là ba giờ mười lăm phút.", "Tôi muốn chạy", "Đồng ý", "tệp", "Ơ"]:
            self.assertEqual(detect_language(text), "vi", text)


class ApprovalPromptTests(VoiceTestCase):
    def test_every_new_approval_request_is_announced_even_if_it_looks_the_same(self):
        chat = FakeChat()
        chat.busy = True
        first = {"calls": [{"name": "terminal", "args": {"command": "ls"}}]}
        chat.pending_approval = first
        self.build(audios=[SPEECH, SPEECH], texts=["yes", "yes"], chat=chat)
        thread = threading.Thread(target=self.assistant.run, daemon=True)
        thread.start()

        self.assertTrue(wait_for(lambda: chat.submitted == ["y"]))
        self.assertIs(self.assistant._asked, first, "held by reference, so its id cannot be reused")

        chat.pending_approval = None
        del first
        chat.pending_approval = {"calls": [{"name": "terminal", "args": {"command": "ls"}}]}  # equal, but a new request
        self.assertTrue(wait_for(lambda: chat.submitted == ["y", "y"]))
        self.assertEqual(len(self.speaker.said), 2, "the second request must be spoken too")
        self.assistant.request_stop()
        thread.join(5)

    def test_the_same_request_is_announced_once(self):
        chat = FakeChat()
        chat.busy = True
        chat.pending_approval = {"calls": [{"name": "terminal", "args": {"command": "ls"}}]}
        self.build(audios=[None], texts=[], chat=chat)  # silence: the approval is declined
        self.run_until(lambda: chat.submitted == ["n"])
        self.assertEqual(len(self.speaker.said), 1)


class CancelledSpeechTests(VoiceTestCase):
    def test_speech_taken_from_the_queue_just_before_a_stop_is_not_spoken(self):
        self.build()
        self.assistant.state = S.SPEAKING
        self.assistant._speech = ("this was queued before the stop", False, self.assistant._speech_epoch)
        self.assistant._drop_pending_speech()  # the stop arrives before say() starts
        self.assistant._speak()

        self.assertEqual(self.speaker.said, [])
        self.assertIs(self.assistant.state, S.RETURN_TO_STANDBY)

    def test_a_cancelled_approval_prompt_does_not_leave_the_assistant_waiting_for_an_answer(self):
        self.build()
        self.assistant.state = S.SPEAKING
        self.assistant._speech = ("Do you allow it?", True, self.assistant._speech_epoch)
        self.assistant._drop_pending_speech()
        self.assistant._speak()

        self.assertEqual(self.speaker.said, [])
        self.assertIs(self.assistant.state, S.RETURN_TO_STANDBY, "no LISTENING for a question nobody asked")
        self.assertFalse(self.assistant._expecting_answer)

    def test_speech_queued_after_a_stop_is_spoken_normally(self):
        self.build()
        self.assistant._drop_pending_speech()
        self.assistant.state = S.SPEAKING
        self.assistant._speech = ("fresh reply", False, self.assistant._speech_epoch)
        self.assistant._speak()
        self.assertEqual(self.speaker.said, ["fresh reply"])


class ShutdownTests(VoiceTestCase):
    def test_a_failing_speaker_close_does_not_leave_the_microphone_open(self):
        self.build()

        def boom():
            raise RuntimeError("audio driver crashed")

        self.speaker.close = boom
        thread = threading.Thread(target=self.assistant.run, daemon=True)
        thread.start()
        self.assertTrue(wait_for(lambda: self.mic.start_calls == 1))
        self.assistant.request_stop()
        thread.join(5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(self.mic.closed, 1, "the microphone must be released even though the speaker failed")

    def test_a_failing_microphone_close_does_not_raise_out_of_run(self):
        self.build()
        self.mic.close = lambda: (_ for _ in ()).throw(OSError("device gone"))
        thread = threading.Thread(target=self.assistant.run, daemon=True)
        thread.start()
        wait_for(lambda: self.mic.start_calls == 1)
        self.assistant.request_stop()
        thread.join(5)
        self.assertFalse(thread.is_alive())


class RecoveryTests(VoiceTestCase):
    def test_a_failure_while_returning_to_standby_does_not_end_voice_mode(self):
        calls = []

        def broken_record(mic, preroll=None, listen_timeout=10, max_duration=60):
            calls.append(1)

            if len(calls) == 1:
                raise RuntimeError("recorder crashed")  # not handled inside _listen

            return SPEECH

        self.build(frames=[WAKE, WAKE], texts=["Jarvis, still alive"])
        self.recorder.record = broken_record
        original = self.assistant._to_standby
        failures = []

        def flaky_to_standby():
            if not failures:
                failures.append(1)
                raise RuntimeError("could not reach standby")

            original()

        self.assistant._to_standby = flaky_to_standby
        self.run_until(lambda: self.chat.submitted)
        self.assertEqual(self.chat.submitted, ["still alive"])
        self.assertEqual(failures, [1])

    def test_declining_an_approval_survives_a_failing_chat(self):
        class BrokenChat(FakeChat):
            def submit(self, line):
                self.submitted.append(line)
                raise RuntimeError("chat is broken")

        chat = BrokenChat()
        chat.busy = True
        chat.pending_approval = {"calls": [{"name": "terminal", "args": {"command": "ls"}}]}
        self.build(audios=[None], chat=chat)  # silence -> decline -> submit raises
        self.run_until(lambda: chat.submitted == ["n"])
        self.assertGreaterEqual(self.assistant.cycles, 0)


class FakeStream:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def start(self):
        self.log.append((self.name, "start"))

    def write(self, data):
        self.log.append((self.name, "write"))

    def stop(self):
        self.log.append((self.name, "stop"))

    def abort(self):
        self.log.append((self.name, "abort"))

    def close(self):
        self.log.append((self.name, "close"))


class AudioPlayerTests(unittest.TestCase):
    def setUp(self):
        self.log = []
        names = iter(["s1", "s2", "s3"])
        fake = ModuleType("sounddevice")
        fake.OutputStream = lambda **kwargs: FakeStream(self.log, next(names))  # type: ignore[attr-defined]
        patcher = patch.dict(sys.modules, {"sounddevice": fake})
        patcher.start()
        self.addCleanup(patcher.stop)

    def clip(self, rate):
        return np.ones(rate // 5, dtype=np.int16)  # two chunks

    def test_a_sample_rate_change_lets_the_previous_audio_finish_first(self):
        player = AudioPlayer()
        player.play(self.clip(22050), 22050)
        player.play(self.clip(16000), 16000)  # the beep, at another rate
        self.assertLess(self.log.index(("s1", "stop")), self.log.index(("s1", "close")), "drain before closing")
        self.assertNotIn(("s1", "abort"), self.log)
        self.assertIn(("s2", "start"), self.log)

    def test_the_stream_is_reused_at_the_same_rate(self):
        player = AudioPlayer()
        player.play(self.clip(22050), 22050)
        player.play(self.clip(22050), 22050)
        self.assertEqual([e for e in self.log if e[1] == "start"], [("s1", "start")])

    def test_close_drains_by_default(self):
        player = AudioPlayer()
        player.play(self.clip(16000), 16000)
        player.close()
        self.assertEqual(self.log[-2:], [("s1", "stop"), ("s1", "close")])

    def test_an_interruption_drops_the_buffered_audio(self):
        player = AudioPlayer()
        player.play(self.clip(16000), 16000, should_stop=lambda: True)
        self.assertIn(("s1", "abort"), self.log)
        self.assertNotIn(("s1", "stop"), self.log)
        self.assertEqual(self.log[-1], ("s1", "close"))

    def test_a_playback_error_discards_and_closes(self):
        player = AudioPlayer()
        player.play(self.clip(16000), 16000)
        stream = player._stream

        def broken_write(data):
            raise OSError("device unplugged")

        stream.write = broken_write  # type: ignore[union-attr]

        with self.assertRaises(OSError):
            player.play(self.clip(16000), 16000)

        self.assertIsNone(player._stream)
        self.assertIn(("s1", "abort"), self.log)

    def test_a_failing_close_is_logged_not_raised(self):
        player = AudioPlayer()
        player.play(self.clip(16000), 16000)
        player._stream.stop = lambda: (_ for _ in ()).throw(OSError("x"))  # type: ignore[union-attr]
        player.close()
        self.assertIsNone(player._stream)


class FakeDownload(io.BytesIO):
    def __init__(self, data, length):
        super().__init__(data)
        self.headers = {"Content-Length": str(length) if length is not None else ""}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name) / "speaker"

    def download(self, data, length="same", timeout_seen=None):
        length = len(data) if length == "same" else length
        seen = timeout_seen if timeout_seen is not None else []

        def fake_urlopen(url, timeout=None):
            seen.append(timeout)
            return FakeDownload(data, length)

        with patch.object(speaker_module.urllib.request, "urlopen", fake_urlopen):
            return speaker_module.SpeakerEmbedder._ensure_model(self.directory)

    def test_a_complete_download_is_stored(self):
        data = b"x" * (speaker_module.MIN_MODEL_BYTES + 10)
        model = self.download(data)
        self.assertEqual(model.read_bytes(), data)
        self.assertEqual([p.name for p in self.directory.iterdir()], [speaker_module.MODEL_FILE], "no .part files left")

    def test_the_download_has_a_timeout(self):
        seen = []
        self.download(b"x" * (speaker_module.MIN_MODEL_BYTES + 1), timeout_seen=seen)
        self.assertEqual(seen, [speaker_module.DOWNLOAD_TIMEOUT])

    def test_a_truncated_download_is_rejected_and_cleaned_up(self):
        data = b"x" * (speaker_module.MIN_MODEL_BYTES + 10)

        with self.assertRaises(RuntimeError):
            self.download(data, length=len(data) + 1000)

        self.assertEqual(list(self.directory.iterdir()), [])

    def test_a_tiny_error_page_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.download(b"<html>404 not found</html>")

        self.assertFalse((self.directory / speaker_module.MODEL_FILE).exists())

    def test_a_server_that_gives_no_length_is_accepted_if_the_file_is_big_enough(self):
        model = self.download(b"x" * (speaker_module.MIN_MODEL_BYTES + 5), length=None)
        self.assertTrue(model.is_file())

    def test_an_existing_good_model_is_not_downloaded_again(self):
        self.directory.mkdir(parents=True)
        (self.directory / speaker_module.MODEL_FILE).write_bytes(b"y" * (speaker_module.MIN_MODEL_BYTES + 1))

        with patch.object(speaker_module.urllib.request, "urlopen", side_effect=AssertionError("no download wanted")):
            speaker_module.SpeakerEmbedder._ensure_model(self.directory)


if __name__ == "__main__":
    unittest.main()
