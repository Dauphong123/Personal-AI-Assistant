"""F036: each web session has its own model; the page lists and switches models over the API."""

import threading
import unittest
from unittest import mock

from tests import stubs

stubs.install()

import backend.server  # noqa: E402
from backend.session import ChatSession  # noqa: E402
from model_choice import ModelChoice  # noqa: E402
from tests.test_backend_server import ServerCase  # noqa: E402
from tests.test_backend_session import wait_for  # noqa: E402

MODELS = ["qwen3:8b", "qwen3:4b", "claude-code:opus"]


def recording(seen: list):
    def run(app, text, config, on_text, approve, cancel=None, **kwargs):
        seen.append(kwargs.get("model"))
        on_text("ok")
        return False

    return run


class SessionModelTests(unittest.TestCase):
    def test_new_session_starts_on_the_given_default(self):
        session = ChatSession(app=None, run=recording([]), choice=ModelChoice("qwen3:4b", None))
        info = session.info()
        self.assertEqual(info["model"], "qwen3:4b")
        self.assertIsNone(info["effort"])

    def test_set_model_is_used_by_the_next_turn_and_announced(self):
        seen: list = []
        session = ChatSession(app=None, run=recording(seen))
        choice = ModelChoice("claude-code:opus", "high")

        self.assertTrue(session.set_model(choice))
        session.send("hi")
        events = wait_for(session, "done")

        self.assertEqual(seen, [choice])
        model_events = [e for e in events if e["type"] == "model"]
        self.assertEqual(model_events[0]["model"], "claude-code:opus")
        self.assertEqual(model_events[0]["effort"], "high")
        self.assertIn("Anthropic", model_events[0]["text"])
        self.assertEqual(session.info()["effort"], "high")

    def test_local_model_has_no_privacy_text(self):
        session = ChatSession(app=None, run=recording([]))
        session.set_model(ModelChoice("qwen3:4b", None))
        [event] = [e for e in session.events_after(0) if e["type"] == "model"]
        self.assertEqual(event["text"], "")

    def test_no_switch_while_a_turn_runs(self):
        gate = threading.Event()

        def slow(app, text, config, on_text, approve, cancel=None, **kwargs):
            gate.wait(5)
            return False

        session = ChatSession(app=None, run=slow)
        session.send("hi")
        self.assertFalse(session.set_model(ModelChoice("qwen3:4b", None)))
        gate.set()
        wait_for(session, "done")


class ModelRouteTests(ServerCase):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(backend.server, "available_models", lambda: list(MODELS))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_models_are_listed(self):
        status, data = self.api("GET", "/api/models")
        self.assertEqual(status, 200)
        self.assertEqual(data["models"], MODELS)
        self.assertEqual(data["efforts"], ["low", "medium", "high", "xhigh", "max"])
        self.assertIn("model", data["default"])

    def test_models_need_the_token(self):
        status, _ = self.api("GET", "/api/models", token=False)
        self.assertEqual(status, 401)

    def test_switch_model(self):
        sid = self.new_session()
        status, data = self.api("POST", f"/api/sessions/{sid}/model", {"model": "claude-code:opus", "effort": "low"})
        self.assertEqual(status, 200)
        self.assertEqual(data["session"]["model"], "claude-code:opus")
        self.assertEqual(data["session"]["effort"], "low")
        self.assertEqual(self.session(sid).choice, ModelChoice("claude-code:opus", "low"))

    def test_bad_model_is_a_400(self):
        sid = self.new_session()
        for body in [{"model": "claude-code:gpt"}, {"model": 5}, {"model": "qwen3:4b", "effort": "high"}, {}]:
            with self.subTest(body=body):
                status, data = self.api("POST", f"/api/sessions/{sid}/model", body)
                self.assertEqual(status, 400)
                self.assertIn("error", data)

    def test_unknown_session_is_a_404(self):
        status, _ = self.api("POST", "/api/sessions/0123456789ab/model", {"model": "qwen3:4b"})
        self.assertEqual(status, 404)

    def test_busy_session_is_a_409(self):
        sid = self.new_session()
        with mock.patch.object(ChatSession, "set_model", return_value=False):
            status, _ = self.api("POST", f"/api/sessions/{sid}/model", {"model": "qwen3:4b"})
        self.assertEqual(status, 409)

    def test_handshake_offers_models(self):
        status, data = self.api("POST", "/api/handshake", {"protocol": backend.server.PROTOCOL, "client": "test"})
        self.assertEqual(status, 200)
        self.assertTrue(data["features"]["models"])


if __name__ == "__main__":
    unittest.main()
