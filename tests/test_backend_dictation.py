"""Web dictation (F022): ?dictate=1 on /api/voice/transcribe adds the cleaned prompt, any
hands-free command and the approval answer it would be; without it the answer is unchanged."""

import json
import unittest

from tests import stubs

stubs.install()

from backend.voice import dictation  # noqa: E402
from tests.test_backend_server import ServerCase  # noqa: E402
from tests.test_backend_voice import make_voice  # noqa: E402


class DictationTests(unittest.TestCase):
    def test_prompt_is_cleaned(self):
        self.assertEqual(dictation("um, open the the readme"), {"prompt": "Open the readme.", "hands_free": None, "answer": "n"})

    def test_hands_free_commands(self):
        self.assertEqual(dictation("hands free")["hands_free"], "on")
        self.assertEqual(dictation("Stop listening.")["hands_free"], "off")

    def test_chat_commands_stay_plain(self):
        self.assertEqual(dictation("Stop.")["prompt"], "stop")

    def test_approval_answers(self):
        self.assertEqual(dictation("Yes.")["answer"], "y")
        self.assertEqual(dictation("vâng")["answer"], "y")
        self.assertEqual(dictation("yes, no wait")["answer"], "n")
        self.assertEqual(dictation("stop")["answer"], "stop")

    def test_nothing_heard(self):
        self.assertEqual(dictation(""), {"prompt": "", "hands_free": None, "answer": "n"})


class DictateRouteTests(ServerCase):
    def setUp(self):
        super().setUp()
        self.voice, self.transcriber, _ = make_voice()
        self.server.voice = self.voice

    def post(self, path):
        return self.request("POST", path, raw=b"opus", headers={"Content-Type": "audio/webm"})

    def test_dictate_adds_the_prompt(self):
        status, _, body = self.post("/api/voice/transcribe?dictate=1")
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(body), {"text": "hello from the mic", "prompt": "Hello from the mic.", "hands_free": None, "answer": "n"}
        )

    def test_plain_transcribe_is_unchanged(self):
        status, _, body = self.post("/api/voice/transcribe")
        self.assertEqual(json.loads(body), {"text": "hello from the mic"})


if __name__ == "__main__":
    unittest.main()
