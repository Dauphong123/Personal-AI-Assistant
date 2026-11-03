"""Vietnamese support: language detection, stop/approval phrases, wake-word forms,
voice selection, configuration, and Vietnamese typed and spoken approvals."""

import re
import unittest
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tests import stubs

stubs.install()

import chat_loop  # noqa: E402
import tools  # noqa: E402
from chat_loop import ChatLoop  # noqa: E402
from jobs import JobManager  # noqa: E402
from language import detect_language, normalize, say  # noqa: E402
from tests.test_chat_loop import FakeTurns, wait_for  # noqa: E402
from tests.test_voice import FakeChat, VoiceTestCase  # noqa: E402
from tests.test_voice_parts import FakePlayer, FakeSynth  # noqa: E402
from turns import is_stop_request  # noqa: E402
from voice import text as vtext  # noqa: E402
from voice.assistant import describe_approval  # noqa: E402
from voice.config import VoiceConfig, VoiceConfigError  # noqa: E402
from voice.enroll import PHRASES, PHRASES_VI, phrases_for  # noqa: E402
from voice.stt import FasterWhisperTranscriber  # noqa: E402
from voice.tts import LazySynthesizer, Speaker  # noqa: E402
from tests.test_voice import SPEECH, WAKE  # noqa: E402


class DetectLanguageTests(unittest.TestCase):
    def test_vietnamese_is_recognized_from_its_letters(self):
        for text in [
            "Xin chào, bạn khỏe không?",
            "Hãy chạy các bài kiểm tra và cho tôi biết kết quả.",
            "Đồng ý",
            "tệp",
        ]:
            self.assertEqual(detect_language(text), "vi", text)

    def test_other_text_is_english(self):
        for text in ["run the tests", "café au lait", "", "123 !!!", "xin chao (no accents)"]:
            self.assertEqual(detect_language(text), "en", text)

    def test_decomposed_unicode_is_handled(self):
        decomposed = unicodedata.normalize("NFD", "Xin chào, bạn có khỏe không?")
        self.assertNotEqual(decomposed, "Xin chào, bạn có khỏe không?")
        self.assertEqual(detect_language(decomposed), "vi")
        self.assertEqual(normalize(decomposed), "Xin chào, bạn có khỏe không?")

    def test_localized_phrases_fall_back_to_english(self):
        self.assertIn("Xin lỗi", say("apology", "vi"))
        self.assertIn("Sorry", say("apology", "en"))
        self.assertEqual(say("apology", "fr"), say("apology", "en"))


class StopPhraseTests(unittest.TestCase):
    def test_vietnamese_stop_phrases(self):
        for text in [
            "dừng", "Dừng lại!", "dừng lại ngay", "hủy", "huỷ bỏ", "hủy bỏ", "ngừng", "thôi",
            "thôi đi", "bỏ đi", "đừng làm nữa", "làm ơn dừng lại", "khoan, dừng", "dung lai",
            "huy bo", "dừng việc này lại",
        ]:  # fmt: skip
            self.assertTrue(is_stop_request(text), text)

    def test_requests_that_merely_contain_those_words_are_not_stop(self):
        for text in [
            "dừng chân ở đâu thì đẹp", "hủy đăng ký như thế nào", "thôi việc là gì",
            "dung", "đúng", "hãy dừng container docker lại",
        ]:  # fmt: skip
            self.assertFalse(is_stop_request(text), text)

    def test_decomposed_input_still_matches(self):
        self.assertTrue(is_stop_request(unicodedata.normalize("NFD", "Dừng lại")))


class VoiceTextTests(unittest.TestCase):
    def test_spoken_yes_and_no(self):
        for yes in ["Có.", "Vâng ạ", "Dạ", "Đồng ý, cho phép chạy", "Được rồi", "Có, cho phép"]:
            self.assertEqual(vtext.normalize_answer(yes), "y", yes)

        for no in ["Không", "Không được", "để tôi nghĩ đã", "Khoan đã"]:
            self.assertEqual(vtext.normalize_answer(no), "n", no)

        self.assertEqual(vtext.normalize_answer("Dừng lại"), "stop")

    def test_vietnamese_wake_word_forms(self):
        cases = {
            "Hey Jarvis! Hãy chạy các bài kiểm tra!": "Hãy chạy các bài kiểm tra!",
            "Này Jarvis, mở thư mục": "mở thư mục",
            "Ê Jarvis liệt kê các tệp": "liệt kê các tệp",
            "Xin chào Jarvis, mấy giờ rồi?": "mấy giờ rồi?",
            "Jarvis ơi, giúp tôi với": "giúp tôi với",
            "Jarvis.": "",
        }
        for heard, expected in cases.items():
            self.assertEqual(vtext.strip_wake_word(heard, "hey_jarvis"), expected, heard)

    def test_vietnamese_recognizer_noise_is_filler(self):
        for noise in [
            "ừm", "Cảm ơn các bạn", "Hẹn gặp lại", "À.",
            "Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn",
        ]:  # fmt: skip
            self.assertTrue(vtext.is_filler(noise), noise)

        for request in ["Hãy chạy bài kiểm tra", "Cảm ơn bạn nhiều vì đã giúp tôi đọc tệp này"]:
            self.assertFalse(vtext.is_filler(request), request)

    def test_vietnamese_reply_is_cleaned_and_split_for_speech(self):
        spoken = vtext.clean_for_speech("**Xin chào!** Đây là `ls`. Bạn cần gì không?", 500)
        self.assertEqual(vtext.split_sentences(spoken), ["Xin chào!", "Đây là ls.", "Bạn cần gì không?"])


class VoiceChoiceTests(unittest.TestCase):
    def test_replies_are_spoken_with_the_voice_of_their_language(self):
        english, vietnamese = FakeSynth(), FakeSynth()
        speaker = Speaker(english, FakePlayer(), synthesizers={"vi": vietnamese})
        speaker.say("Hello there. How are you?")
        speaker.say("Xin chào. Bạn khỏe không?")
        self.assertEqual(english.calls, ["Hello there.", "How are you?"])
        self.assertEqual(vietnamese.calls, ["Xin chào.", "Bạn khỏe không?"])

    def test_without_a_vietnamese_voice_the_default_is_used(self):
        english = FakeSynth()
        Speaker(english, FakePlayer()).say("Xin chào.")
        self.assertEqual(english.calls, ["Xin chào."])

    def test_a_voice_is_loaded_only_when_first_used(self):
        loaded = []

        def factory():
            loaded.append(1)
            return FakeSynth()

        lazy = LazySynthesizer(factory)
        self.assertEqual(loaded, [])
        lazy.synthesize("one")
        lazy.synthesize("two")
        self.assertEqual(loaded, [1])


class ConfigTests(unittest.TestCase):
    def test_non_english_languages_default_to_the_more_accurate_whisper_model(self):
        self.assertEqual(VoiceConfig.from_env({}).stt_model, "base")
        self.assertEqual(VoiceConfig.from_env({"SPEECH_LANGUAGE": "vi"}).stt_model, "small")
        self.assertEqual(VoiceConfig.from_env({"SPEECH_LANGUAGE": "auto"}).stt_model, "small")

    def test_an_explicit_model_wins(self):
        env = {"SPEECH_LANGUAGE": "vi", "SPEECH_RECOGNITION_MODEL": "medium"}
        self.assertEqual(VoiceConfig.from_env(env).stt_model, "medium")

    def test_language_values(self):
        self.assertEqual(VoiceConfig.from_env({"SPEECH_LANGUAGE": "VI"}).stt_language, "vi")
        self.assertEqual(VoiceConfig.from_env({"SPEECH_LANGUAGE": "Auto"}).stt_language, "auto")

        for bad in ["english", "v1", "vi-VN", "a"]:
            with self.assertRaises(VoiceConfigError, msg=bad):
                VoiceConfig.from_env({"SPEECH_LANGUAGE": bad})

    def test_several_languages_can_be_listed(self):
        config = VoiceConfig.from_env({"SPEECH_LANGUAGE": "vi, EN"})
        self.assertEqual(config.stt_language, "vi,en")
        self.assertEqual(config.stt_languages, ("vi", "en"))
        self.assertEqual(config.stt_model, "small")
        self.assertEqual(VoiceConfig.from_env({"SPEECH_LANGUAGE": "vi"}).stt_languages, ("vi",))
        self.assertEqual(VoiceConfig.from_env({"SPEECH_LANGUAGE": "auto"}).stt_languages, ())

        for bad in ["auto,en", "vi,", ",en", "vi,english"]:
            with self.assertRaises(VoiceConfigError, msg=bad):
                VoiceConfig.from_env({"SPEECH_LANGUAGE": bad})

    def test_vietnamese_voice_is_configurable(self):
        self.assertEqual(VoiceConfig.from_env({}).tts_voice_vi, "vi_VN-vais1000-medium")
        self.assertEqual(VoiceConfig.from_env({"TTS_VOICE_VI": "vi_VN-x"}).tts_voice_vi, "vi_VN-x")

    def test_enrollment_phrases_follow_the_language(self):
        self.assertIs(phrases_for("vi"), PHRASES_VI)
        self.assertIs(phrases_for("en"), PHRASES)
        self.assertIs(phrases_for("auto"), PHRASES)
        self.assertTrue(all(detect_language(p) == "vi" for p in PHRASES_VI))


class FakeWhisper:
    """Stands in for faster-whisper's model: reports language odds, records what it was asked."""

    def __init__(self, odds):
        self.odds = odds
        self.asked_language = "unset"
        self.detections = 0

    def detect_language(self, samples):
        self.detections += 1
        return "xx", 0.9, self.odds

    def transcribe(self, samples, language=None, **kwargs):
        self.asked_language = language
        segment = SimpleNamespace(text=" xin chào ", no_speech_prob=0.0, avg_logprob=0.0)
        return iter([segment]), None


class LanguageChoiceTests(unittest.TestCase):
    AUDIO = np.zeros(16000, dtype=np.int16)

    def transcriber(self, languages, odds=()):
        stt = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)  # skip loading the model
        stt.candidates = languages
        stt.language = languages[0] if len(languages) == 1 else None
        stt._model = FakeWhisper(list(odds))
        return stt

    def test_one_language_is_forced_without_detecting(self):
        stt = self.transcriber(("vi",))
        self.assertEqual(stt.transcribe(self.AUDIO), "xin chào")
        self.assertEqual(stt._model.asked_language, "vi")
        self.assertEqual(stt._model.detections, 0)

    def test_with_several_languages_the_likeliest_of_them_is_used(self):
        # Whisper thinks Thai is likeliest overall, but only Vietnamese and English are allowed.
        odds = [("th", 0.5), ("en", 0.1), ("vi", 0.3)]
        stt = self.transcriber(("vi", "en"), odds)
        stt.transcribe(self.AUDIO)
        self.assertEqual(stt._model.asked_language, "vi")

        stt = self.transcriber(("vi", "en"), [("en", 0.8), ("vi", 0.1)])
        stt.transcribe(self.AUDIO)
        self.assertEqual(stt._model.asked_language, "en")

    def test_a_candidate_missing_from_the_odds_counts_as_impossible(self):
        stt = self.transcriber(("vi", "en"), [("en", 0.01)])
        stt.transcribe(self.AUDIO)
        self.assertEqual(stt._model.asked_language, "en")

    def test_auto_lets_whisper_decide(self):
        stt = self.transcriber(())
        stt.transcribe(self.AUDIO)
        self.assertIsNone(stt._model.asked_language)
        self.assertEqual(stt._model.detections, 0)

    def test_audio_that_is_too_short_is_ignored_before_detecting(self):
        stt = self.transcriber(("vi", "en"), [("vi", 1.0)])
        self.assertEqual(stt.transcribe(np.zeros(100, dtype=np.int16)), "")
        self.assertEqual(stt._model.detections, 0)


class TypedChatTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeTurns()
        self.fake.hold_first = False
        self.fake.ask_approval = True
        patcher = patch.object(chat_loop, "run_turn", self.fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.out = []
        self.chat = ChatLoop(None, {}, JobManager(), write=self.out.append)
        self.chat.start()
        self.addCleanup(self.chat.close)

    def answer(self, line):
        self.fake.approvals.clear()
        self.chat.submit("a")
        self.assertTrue(wait_for(lambda: self.chat._approval is not None))
        self.chat.submit(line)
        self.assertTrue(wait_for(lambda: not self.chat.busy))
        return self.fake.approvals[0]

    def test_vietnamese_yes_approves(self):
        for line in ["có", "Có.", "vâng", "Đồng ý!", "dong y", unicodedata.normalize("NFD", "có")]:
            self.assertTrue(self.answer(line), line)

    def test_other_answers_decline(self):
        for line in ["không", "Không được", "hmm"]:
            self.assertFalse(self.answer(line), line)

    def test_vietnamese_stop_while_working_cancels(self):
        self.fake.ask_approval = False
        self.fake.hold_first = True
        self.chat.submit("chạy bài kiểm tra")
        self.assertTrue(self.fake.started.wait(5))
        self.chat.submit("dừng lại")
        self.assertTrue(wait_for(lambda: not self.chat.busy))
        self.assertIn("[Stopped.]", "".join(self.out))


class ApologyLanguageTests(unittest.TestCase):
    def failing_chat(self):
        replies = []
        chat = ChatLoop(None, {}, JobManager(), write=lambda s: None, on_reply=replies.append)

        def boom(*args, **kwargs):
            raise RuntimeError("model down")

        patcher = patch.object(chat_loop, "run_turn", boom)
        patcher.start()
        self.addCleanup(patcher.stop)
        chat.start()
        self.addCleanup(chat.close)
        return chat, replies

    def test_the_apology_matches_the_language_of_the_request(self):
        chat, replies = self.failing_chat()
        chat.submit("Hãy liệt kê các tệp")
        chat.submit("list the files")
        self.assertTrue(wait_for(lambda: len(replies) == 2))
        self.assertIn("Xin lỗi", replies[0])
        self.assertIn("Sorry", replies[1])


class SpokenVietnameseTests(VoiceTestCase):
    REQUEST = {"calls": [{"name": "terminal", "args": {"command": "pytest"}}]}

    def test_approval_prompt_is_spoken_in_vietnamese_when_you_speak_vietnamese(self):
        self.assertIn("Tôi muốn chạy", describe_approval(self.REQUEST, "vi"))
        self.assertIn("đồng ý", describe_approval(self.REQUEST, "vi"))
        self.assertIn("I want to run", describe_approval(self.REQUEST, "en"))

    def run_approval(self, language, answer):
        chat = FakeChat()
        chat.busy = True
        chat.pending_approval = self.REQUEST
        self.build(audios=[SPEECH], texts=[answer], chat=chat, language=language)
        self.run_until(lambda: chat.submitted)
        return chat

    def test_vietnamese_assistant_asks_in_vietnamese_and_understands_the_answer(self):
        chat = self.run_approval("vi", "Có, cho phép.")
        self.assertIn("Tôi muốn chạy", self.speaker.said[0])
        self.assertEqual(chat.submitted, ["y"])

    def test_english_assistant_still_asks_in_english(self):
        chat = self.run_approval("en", "Yes, allow it.")
        self.assertIn("I want to run", self.speaker.said[0])
        self.assertEqual(chat.submitted, ["y"])

    def test_a_vietnamese_command_after_the_wake_word_is_forwarded_clean(self):
        self.build(frames=[WAKE], audios=[SPEECH], texts=["Hey Jarvis! Hãy chạy các bài kiểm tra!"],
                   wake_word="hey_jarvis", language="vi")  # fmt: skip
        self.run_until(lambda: self.chat.submitted)
        self.assertEqual(self.chat.submitted, ["Hãy chạy các bài kiểm tra!"])

    def test_whisper_subscribe_hallucination_is_not_sent_to_the_agent(self):
        junk = "Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn"
        self.build(frames=[WAKE], audios=[SPEECH, SPEECH], texts=[junk, "mấy giờ rồi"], language="vi")
        self.run_until(lambda: self.chat.submitted)
        self.assertEqual(self.chat.submitted, ["mấy giờ rồi"])


class PromptTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parent.parent

    def test_every_tool_the_code_prompt_mentions_exists(self):
        prompt = (self.ROOT / "chains" / "code_chain.py").read_text(encoding="utf-8")
        mentioned = set(re.findall(r"`(\w+)\(\)`", prompt))
        self.assertTrue(mentioned)
        self.assertEqual(mentioned - set(tools.TOOL_BY_NAME), set())

    def test_prompts_ask_for_the_users_language(self):
        for name in ["code_chain", "direct_chain"]:
            source = (self.ROOT / "chains" / f"{name}.py").read_text(encoding="utf-8")
            self.assertIn("language the user writes in", source, name)


if __name__ == "__main__":
    unittest.main()
