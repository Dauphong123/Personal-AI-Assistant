import http.client
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from tests import stubs

stubs.install()

from backend.server import make_server  # noqa: E402
from backend.sessions import SessionManager  # noqa: E402
from tests.test_backend_session import scripted  # noqa: E402

TOKEN = "test-token-123"


class ServerCase(unittest.TestCase):
    run_fn = staticmethod(scripted("Hel", "lo"))
    max_turns = 1

    def setUp(self):
        self.static = tempfile.TemporaryDirectory()
        root = Path(self.static.name)
        (root / "index.html").write_text("<html>page</html>", encoding="utf-8")
        (root / "main.js").write_text("// js", encoding="utf-8")
        (root / "style.css").write_text("/* css */", encoding="utf-8")
        (root / "lib").mkdir()
        (root / "lib" / "api.js").write_text("// api", encoding="utf-8")
        self.manager = SessionManager(app=None, run=self.run_fn, max_turns=self.max_turns)
        self.server = make_server(self.manager, TOKEN, port=0, static_dir=root)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.static.cleanup()

    def request(self, method, path, body=None, headers=None, token=True, host=None, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        hdrs = {"Host": host or f"127.0.0.1:{self.port}"}
        if token:
            hdrs["X-Token"] = TOKEN
        if body is not None:
            raw = json.dumps(body)
            hdrs["Content-Type"] = "application/json"
        hdrs.update(headers or {})
        conn.request(method, path, body=raw, headers=hdrs)
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, resp, data

    def api(self, method, path, body=None, **kwargs):
        """(status, decoded JSON) of an API call."""
        status, _, data = self.request(method, path, body, **kwargs)
        return status, json.loads(data or b"{}")

    def new_session(self, **body) -> str:
        status, data = self.api("POST", "/api/sessions", body)
        self.assertEqual(status, 201)
        return data["session"]["id"]

    def read_events(self, sid, until="done", headers=None, path=None, sink=None):
        """Open a session's SSE stream and collect events up to (and including) one of type `until`.

        With `sink`, events are appended to it as they arrive (and an empty list is returned)."""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        hdrs = {"X-Token": TOKEN}
        hdrs.update(headers or {})
        conn.request("GET", path or f"/api/sessions/{sid}/events", headers=hdrs)
        resp = conn.getresponse()
        self.assertEqual(resp.status, 200)
        self.assertIn("text/event-stream", resp.getheader("Content-Type"))
        events = sink if sink is not None else []
        while True:
            line = resp.fp.readline().decode("utf-8")
            if not line:
                break
            if line.startswith("data: "):
                event = json.loads(line[len("data: "):])
                events.append(event)
                if event["type"] == until:
                    break
        conn.close()
        return [] if sink is not None else events


class LoopbackTests(unittest.TestCase):
    def manager(self):
        return SessionManager(app=None, run=scripted())

    def test_non_loopback_host_is_refused(self):
        with self.assertRaises(ValueError):
            make_server(self.manager(), TOKEN, host="0.0.0.0", port=0)

    def test_non_loopback_needs_explicit_flag(self):
        make_server(self.manager(), TOKEN, host="0.0.0.0", port=0, allow_remote=True).server_close()

    def test_a_busy_port_is_an_error_not_a_silent_second_bind(self):
        # SO_REUSEADDR lets two processes share a port on Windows, so requests could go to either.
        first = make_server(self.manager(), TOKEN, port=0)
        try:
            with self.assertRaises(OSError):
                make_server(self.manager(), TOKEN, port=first.server_address[1])
        finally:
            first.server_close()

    def test_loopback_hosts_are_accepted(self):
        for host in ("127.0.0.1", "localhost"):
            make_server(self.manager(), TOKEN, host=host, port=0).server_close()


class StaticTests(ServerCase):
    def test_page_is_served_without_token_with_strict_csp(self):
        status, resp, body = self.request("GET", "/", token=False)
        self.assertEqual(status, 200)
        self.assertIn(b"page", body)
        self.assertIn("text/html", resp.getheader("Content-Type"))
        csp = resp.getheader("Content-Security-Policy")
        self.assertIn("script-src 'self'", csp)
        self.assertNotIn("unsafe-inline", csp)
        self.assertEqual(resp.getheader("X-Content-Type-Options"), "nosniff")
        self.assertEqual(resp.getheader("Cache-Control"), "no-store")

    def test_built_files_are_served_with_their_type(self):
        status, resp, _ = self.request("GET", "/main.js", token=False)
        self.assertEqual(status, 200)
        self.assertIn("text/javascript", resp.getheader("Content-Type"))
        status, resp, _ = self.request("GET", "/style.css", token=False)
        self.assertIn("text/css", resp.getheader("Content-Type"))
        self.assertEqual(self.request("GET", "/lib/api.js", token=False)[0], 200)

    def test_query_string_is_ignored_for_files(self):
        self.assertEqual(self.request("GET", "/?token=abc", token=False)[0], 200)

    def test_unknown_types_and_missing_files_are_404(self):
        Path(self.static.name, "secret.txt").write_text("x", encoding="utf-8")
        self.assertEqual(self.request("GET", "/secret.txt", token=False)[0], 404)
        self.assertEqual(self.request("GET", "/nope.js", token=False)[0], 404)
        self.assertEqual(self.request("GET", "/lib", token=False)[0], 404)

    def test_no_path_traversal(self):
        outside = Path(self.static.name).parent / "outside-f019.js"
        outside.write_text("x", encoding="utf-8")
        try:
            for path in ("/../outside-f019.js", "/lib/../../outside-f019.js", "/%2e%2e/outside-f019.js",
                         "/lib%2f..%2f..%2foutside-f019.js", "/..\\outside-f019.js", "/.hidden.js"):
                self.assertEqual(self.request("GET", path, token=False)[0], 404, path)
        finally:
            outside.unlink()

    def test_missing_build_explains_how_to_build(self):
        self.server.static_dir = Path(self.static.name) / "absent"
        status, _, body = self.request("GET", "/", token=False)
        self.assertEqual(status, 503)
        self.assertIn(b"npm", body)


class SecurityTests(ServerCase):
    def test_api_requires_the_token(self):
        self.assertEqual(self.request("GET", "/api/sessions", token=False)[0], 401)
        self.assertEqual(self.request("POST", "/api/sessions", {}, token=False)[0], 401)
        self.assertEqual(self.request("POST", "/api/sessions", {}, headers={"X-Token": "wrong"}, token=False)[0], 401)
        self.assertEqual(self.manager.list(), [])

    def test_bearer_token_is_accepted(self):
        status, _, _ = self.request("GET", "/api/sessions", token=False, headers={"Authorization": f"Bearer {TOKEN}"})
        self.assertEqual(status, 200)

    def test_events_require_the_token(self):
        sid = self.new_session()
        self.assertEqual(self.request("GET", f"/api/sessions/{sid}/events", token=False)[0], 401)

    def test_unknown_session_does_not_leak_without_token(self):
        self.assertEqual(self.request("GET", "/api/sessions/000000000000/events", token=False)[0], 401)

    def test_foreign_host_header_is_refused(self):
        # DNS rebinding: a page on evil.example resolving to 127.0.0.1 sends its own Host.
        self.assertEqual(self.request("GET", "/", token=False, host="evil.example")[0], 403)
        self.assertEqual(self.request("POST", "/api/sessions", {}, host="evil.example")[0], 403)

    def test_localhost_host_header_is_accepted(self):
        self.assertEqual(self.request("GET", "/", token=False, host=f"localhost:{self.port}")[0], 200)

    def test_foreign_origin_is_refused(self):
        status, _, _ = self.request("POST", "/api/sessions", {}, headers={"Origin": "http://evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(self.manager.list(), [])

    def test_same_origin_is_accepted(self):
        origin = f"http://127.0.0.1:{self.port}"
        self.assertEqual(self.request("POST", "/api/sessions", {}, headers={"Origin": origin})[0], 201)

    def test_post_must_be_json(self):
        status, _, _ = self.request(
            "POST", "/api/sessions", raw="title=x", headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        self.assertEqual(status, 415)

    def test_oversize_body_is_refused(self):
        sid = self.new_session()
        self.assertEqual(self.request("POST", f"/api/sessions/{sid}/send", {"text": "x" * 70_000})[0], 413)
        self.assertEqual(self.manager.get(sid).events_after(0), [])

    def test_invalid_json_is_a_400(self):
        status, _, _ = self.request("POST", "/api/sessions", raw="{nope", headers={"Content-Type": "application/json"})
        self.assertEqual(status, 400)

    def test_token_is_accepted_in_query_for_the_event_stream_only(self):
        sid = self.new_session()
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", f"/api/sessions/{sid}/events?token={TOKEN}", headers={"Host": f"127.0.0.1:{self.port}"})
        self.assertEqual(conn.getresponse().status, 200)
        conn.close()
        self.assertEqual(self.request("GET", f"/api/sessions?token={TOKEN}", token=False)[0], 401)


class SessionApiTests(ServerCase):
    def test_list_starts_empty(self):
        self.assertEqual(self.api("GET", "/api/sessions"), (200, {"sessions": []}))

    def test_create_and_list(self):
        status, data = self.api("POST", "/api/sessions", {"title": "Plans"})
        self.assertEqual(status, 201)
        self.assertEqual(data["session"]["title"], "Plans")
        self.assertEqual(data["session"]["state"], "idle")
        second = self.new_session()
        _, listing = self.api("GET", "/api/sessions")
        self.assertEqual([s["id"] for s in listing["sessions"]], [data["session"]["id"], second])

    def test_title_must_be_a_string(self):
        self.assertEqual(self.api("POST", "/api/sessions", {"title": 5})[0], 400)

    def test_get_one(self):
        sid = self.new_session()
        status, data = self.api("GET", f"/api/sessions/{sid}")
        self.assertEqual(status, 200)
        self.assertEqual(data["session"]["id"], sid)

    def test_unknown_session_is_404(self):
        for method, path in (
            ("GET", "/api/sessions/000000000000"),
            ("GET", "/api/sessions/000000000000/events"),
            ("POST", "/api/sessions/000000000000/send"),
            ("POST", "/api/sessions/000000000000/delete"),
        ):
            body = {"text": "hi"} if method == "POST" else None
            self.assertEqual(self.request(method, path, body)[0], 404, path)

    def test_malformed_routes_are_404(self):
        sid = self.new_session()
        for path in ("/api/sessions/NOTANID", f"/api/sessions/{sid}/nope", "/api/send", "/api/other"):
            self.assertEqual(self.request("POST", path, {})[0], 404, path)

    def test_session_limit_is_a_429(self):
        self.manager._max_sessions = 1
        self.new_session()
        self.assertEqual(self.api("POST", "/api/sessions", {})[0], 429)

    def test_send_then_stream_events(self):
        sid = self.new_session()
        self.assertEqual(self.api("POST", f"/api/sessions/{sid}/send", {"text": "hi"})[0], 200)
        events = self.read_events(sid)
        self.assertEqual([e["type"] for e in events], ["user", "token", "token", "done"])
        self.assertEqual(events[0]["text"], "hi")

    def test_streams_are_per_session(self):
        a, b = self.new_session(), self.new_session()
        self.api("POST", f"/api/sessions/{a}/send", {"text": "for a"})
        self.read_events(a)
        self.api("POST", f"/api/sessions/{b}/send", {"text": "for b"})
        events = self.read_events(b)
        self.assertEqual(events[0]["text"], "for b")
        self.assertNotIn("for a", json.dumps(events))

    def test_stream_resumes_after_last_event_id(self):
        sid = self.new_session()
        self.api("POST", f"/api/sessions/{sid}/send", {"text": "hi"})
        first = self.read_events(sid)
        resumed = self.read_events(sid, headers={"Last-Event-ID": str(first[0]["id"])})
        self.assertEqual([e["type"] for e in resumed], ["token", "token", "done"])

    def test_after_query_parameter(self):
        sid = self.new_session()
        self.api("POST", f"/api/sessions/{sid}/send", {"text": "hi"})
        first = self.read_events(sid)
        resumed = self.read_events(sid, path=f"/api/sessions/{sid}/events?after={first[-2]['id']}")
        self.assertEqual([e["type"] for e in resumed], ["done"])

    def test_blank_message_is_a_400(self):
        sid = self.new_session()
        for body in ({"text": "  "}, {"nope": 1}, {"text": 5}):
            self.assertEqual(self.api("POST", f"/api/sessions/{sid}/send", body)[0], 400, body)

    def test_approve_without_pending_request_is_a_409(self):
        sid = self.new_session()
        self.assertEqual(self.api("POST", f"/api/sessions/{sid}/approve", {"allow": True})[0], 409)

    def test_approve_needs_a_boolean(self):
        sid = self.new_session()
        self.assertEqual(self.api("POST", f"/api/sessions/{sid}/approve", {"allow": "yes"})[0], 400)

    def test_stop_and_reset(self):
        sid = self.new_session()
        self.assertEqual(self.api("POST", f"/api/sessions/{sid}/stop", {})[0], 200)
        self.assertEqual(self.api("POST", f"/api/sessions/{sid}/reset", {})[0], 200)
        self.assertEqual([e["type"] for e in self.manager.get(sid).events_after(0)], ["reset"])

    def test_rename(self):
        sid = self.new_session()
        status, data = self.api("POST", f"/api/sessions/{sid}/rename", {"title": "Renamed"})
        self.assertEqual(status, 200)
        self.assertEqual(data["session"]["title"], "Renamed")
        self.assertEqual(self.api("POST", f"/api/sessions/{sid}/rename", {"title": " "})[0], 400)
        self.assertEqual(self.api("POST", f"/api/sessions/{sid}/rename", {})[0], 400)

    def test_delete(self):
        sid = self.new_session()
        self.assertEqual(self.api("POST", f"/api/sessions/{sid}/delete", {})[0], 200)
        self.assertEqual(self.api("GET", "/api/sessions"), (200, {"sessions": []}))
        self.assertEqual(self.api("POST", f"/api/sessions/{sid}/delete", {})[0], 404)

    def test_wrong_method_is_405(self):
        sid = self.new_session()
        self.assertEqual(self.request("GET", f"/api/sessions/{sid}/send")[0], 405)
        self.assertEqual(self.request("POST", f"/api/sessions/{sid}/events", {})[0], 405)
        self.assertEqual(self.request("POST", "/", {})[0], 405)
        self.assertEqual(self.request("POST", f"/api/sessions/{sid}", {})[0], 405)


class BusyTests(ServerCase):
    gate = threading.Event()

    @staticmethod
    def run_fn(app, text, config, on_text, approve, cancel=None, **kwargs):
        BusyTests.gate.wait(3)
        return False

    def setUp(self):
        BusyTests.gate.clear()
        super().setUp()

    def tearDown(self):
        BusyTests.gate.set()
        super().tearDown()

    def test_busy_session_refuses_send_reset_and_delete(self):
        sid = self.new_session()
        self.assertEqual(self.api("POST", f"/api/sessions/{sid}/send", {"text": "one"})[0], 200)
        self.assertEqual(self.api("POST", f"/api/sessions/{sid}/send", {"text": "two"})[0], 409)
        self.assertEqual(self.api("POST", f"/api/sessions/{sid}/reset", {})[0], 409)
        self.assertEqual(self.api("POST", f"/api/sessions/{sid}/delete", {})[0], 409)

    def test_other_session_waits_and_list_shows_states(self):
        a, b = self.new_session(), self.new_session()
        self.api("POST", f"/api/sessions/{a}/send", {"text": "one"})
        self.assertEqual(self.api("POST", f"/api/sessions/{b}/send", {"text": "two"})[0], 200)
        self.read_events(b, until="waiting")
        _, listing = self.api("GET", "/api/sessions")
        states = {s["id"]: s["state"] for s in listing["sessions"]}
        self.assertEqual(states, {a: "running", b: "waiting"})
        BusyTests.gate.set()
        self.read_events(b, until="done")


class StreamEndTests(ServerCase):
    def test_stream_ends_when_session_is_deleted(self):
        sid = self.new_session()
        self.api("POST", f"/api/sessions/{sid}/send", {"text": "hi"})
        seen = []

        def listen():
            seen.extend(self.read_events(sid, until="gone", sink=seen))

        listener = threading.Thread(target=listen, daemon=True)
        listener.start()
        deadline = time.monotonic() + 5
        while not any(e["type"] == "done" for e in seen) and time.monotonic() < deadline:
            time.sleep(0.02)  # the stream is open once it has delivered the turn
        self.assertEqual(self.manager.delete(sid), "deleted")
        listener.join(5)
        self.assertFalse(listener.is_alive())
        self.assertEqual(seen[-1]["type"], "gone")


if __name__ == "__main__":
    unittest.main()
