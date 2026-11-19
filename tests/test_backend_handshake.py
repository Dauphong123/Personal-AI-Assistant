import unittest

from tests import stubs

stubs.install()

from backend.server import PROTOCOL, make_server  # noqa: E402
from backend.sessions import SessionManager  # noqa: E402
from tests.test_backend_server import TOKEN, ServerCase  # noqa: E402
from tests.test_backend_session import scripted  # noqa: E402


class HandshakeTests(ServerCase):
    def hello(self, body=None, **kwargs):
        return self.api("POST", "/api/handshake", {"protocol": PROTOCOL, "client": "webui"} if body is None else body, **kwargs)

    def test_protocol_is_a_positive_integer(self):
        self.assertIsInstance(PROTOCOL, int)
        self.assertGreaterEqual(PROTOCOL, 1)

    def test_matching_protocol_gets_server_description(self):
        status, data = self.hello()
        self.assertEqual(status, 200)
        self.assertEqual(data["protocol"], PROTOCOL)
        self.assertEqual(data["server"], "coding-assistant")
        self.assertEqual(data["features"]["sessions"], True)
        self.assertEqual(data["features"]["notices"], False)  # not wired in this test server
        self.assertEqual(data["features"]["voice"], {"transcribe": False, "speak": False})
        self.assertEqual(data["limits"]["max_turns"], 1)
        self.assertEqual(data["limits"]["max_sessions"], 50)
        self.assertEqual(data["limits"]["max_body"], 64 * 1024)

    def test_other_protocol_is_a_409_naming_ours(self):
        status, data = self.hello({"protocol": PROTOCOL + 1})
        self.assertEqual(status, 409)
        self.assertEqual(data["protocol"], PROTOCOL)
        self.assertIn("npm", data["error"])  # tells how to rebuild the frontend

    def test_protocol_must_be_an_integer(self):
        for body in ({}, {"protocol": "1"}, {"protocol": 1.5}, {"protocol": True}):
            self.assertEqual(self.hello(body)[0], 400, body)

    def test_needs_the_token(self):
        self.assertEqual(self.hello(token=False)[0], 401)

    def test_get_is_not_allowed(self):
        self.assertEqual(self.request("GET", "/api/handshake")[0], 405)

    def test_frontend_speaks_the_same_protocol(self):
        import re
        from pathlib import Path

        types = (Path(__file__).resolve().parent.parent / "frontend" / "src" / "types.ts").read_text(encoding="utf-8")
        match = re.search(r"export const PROTOCOL = (\d+);", types)
        assert match is not None, "PROTOCOL not found in types.ts"
        self.assertEqual(int(match.group(1)), PROTOCOL)


class FeatureTests(unittest.TestCase):
    def serve(self, **kwargs):
        server = make_server(SessionManager(app=None, run=scripted()), TOKEN, port=0, **kwargs)
        self.addCleanup(server.server_close)
        return server

    def test_features_follow_what_the_server_was_given(self):
        class Voice:
            can_transcribe = True
            can_speak = False

        server = self.serve(notices=True, voice=Voice())
        features = server.describe()["features"]
        self.assertEqual(features["notices"], True)
        self.assertEqual(features["voice"], {"transcribe": True, "speak": False})


if __name__ == "__main__":
    unittest.main()
