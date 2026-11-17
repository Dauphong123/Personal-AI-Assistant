import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests import stubs

stubs.install()

import backend.__main__ as entry  # noqa: E402
from backend.sessions import SessionManager  # noqa: E402


class FakeServer:
    def __init__(self, interrupt=True):
        self.server_address = ("127.0.0.1", 8123)
        self.interrupt = interrupt
        self.closed = False

    def serve_forever(self):
        if self.interrupt:
            raise KeyboardInterrupt

    def server_close(self):
        self.closed = True


class ParseArgsTests(unittest.TestCase):
    def test_defaults_are_loopback_and_8000(self):
        args = entry.parse_args([])
        self.assertEqual((args.host, args.port, args.allow_remote), ("127.0.0.1", 8000, False))

    def test_options(self):
        args = entry.parse_args(["--host", "localhost", "--port", "9001", "--allow-remote"])
        self.assertEqual((args.host, args.port, args.allow_remote), ("localhost", 9001, True))

    def test_max_turns_defaults_to_one(self):
        self.assertEqual(entry.parse_args([]).max_turns, 1)
        self.assertEqual(entry.parse_args(["--max-turns", "3"]).max_turns, 3)


class UrlTests(unittest.TestCase):
    def test_url_carries_the_token(self):
        self.assertEqual(entry.build_url("127.0.0.1", 8000, "abc"), "http://127.0.0.1:8000/?token=abc")

    def test_wildcard_host_is_shown_as_localhost(self):
        self.assertEqual(entry.build_url("0.0.0.0", 8000, "abc"), "http://localhost:8000/?token=abc")


class MainTests(unittest.TestCase):
    def run_main(self, argv, server=None, make_error=None, built=True):
        out = io.StringIO()
        build_app = mock.Mock(return_value=SimpleNamespace())
        cleanup = mock.Mock()
        make = mock.Mock(side_effect=make_error, return_value=server)
        connect_jobs = mock.Mock()

        with (
            mock.patch.object(entry, "build_app", build_app),
            mock.patch.object(entry, "cleanup", cleanup),
            mock.patch.object(entry, "make_server", make),
            mock.patch.object(entry, "connect_jobs", connect_jobs),
            mock.patch.object(entry, "frontend_built", mock.Mock(return_value=built)),
            contextlib.redirect_stdout(out),
        ):
            try:
                entry.main(argv)
                error = None
            except SystemExit as exc:
                error = exc

        return SimpleNamespace(
            out=out.getvalue(), build_app=build_app, cleanup=cleanup, make=make, connect_jobs=connect_jobs, error=error
        )

    def test_remote_host_is_refused_before_loading_anything(self):
        result = self.run_main(["--host", "0.0.0.0"])
        self.assertIsNotNone(result.error)
        self.assertIn("--allow-remote", str(result.error))
        result.build_app.assert_not_called()
        result.make.assert_not_called()

    def test_remote_host_with_flag_is_allowed(self):
        result = self.run_main(["--host", "0.0.0.0", "--allow-remote"], server=FakeServer())
        self.assertIsNone(result.error)
        self.assertEqual(result.make.call_args.kwargs["allow_remote"], True)

    def test_prints_the_url_with_a_token_and_cleans_up_on_ctrl_c(self):
        server = FakeServer()
        result = self.run_main([], server=server)
        self.assertIsNone(result.error)
        self.assertRegex(result.out, r"http://127\.0\.0\.1:8123/\?token=[\w-]{20,}")
        self.assertTrue(server.closed)
        result.cleanup.assert_called_once()

    def test_each_launch_uses_a_new_token(self):
        tokens = set()
        for _ in range(2):
            result = self.run_main([], server=FakeServer())
            tokens.add(result.make.call_args.args[1])
        self.assertEqual(len(tokens), 2)

    def test_busy_port_gives_a_clear_message_and_cleans_up(self):
        result = self.run_main(["--port", "8000"], make_error=OSError("address in use"))
        self.assertIsNotNone(result.error)
        self.assertIn("8000", str(result.error))
        result.cleanup.assert_called_once()

    def test_unbuilt_frontend_is_refused_before_loading_anything(self):
        result = self.run_main([], built=False)
        self.assertIsNotNone(result.error)
        self.assertIn("npm", str(result.error))
        result.build_app.assert_not_called()

    def test_server_gets_a_session_manager_with_max_turns(self):
        result = self.run_main(["--max-turns", "2"], server=FakeServer())
        manager = result.make.call_args.args[0]
        self.assertIsInstance(manager, SessionManager)
        self.assertIs(manager.app, result.build_app.return_value)
        self.assertEqual(manager._slot._value, 2)

    def test_job_notices_are_connected_to_the_sessions(self):
        result = self.run_main([], server=FakeServer())
        manager = result.make.call_args.args[0]
        result.connect_jobs.assert_called_once_with(manager)
        self.assertEqual(result.make.call_args.kwargs["notices"], True)

    def test_voice_is_off_by_default(self):
        result = self.run_main([], server=FakeServer())
        self.assertIsNone(result.make.call_args.kwargs["voice"])

    def test_voice_flag_gives_the_server_a_web_voice(self):
        fake = SimpleNamespace(can_transcribe=True, can_speak=True)
        with mock.patch.object(entry, "build_voice", mock.Mock(return_value=fake)) as build_voice:
            result = self.run_main(["--voice"], server=FakeServer())
        build_voice.assert_called_once()
        self.assertIs(result.make.call_args.kwargs["voice"], fake)

    def test_voice_flag_without_packages_says_so(self):
        fake = SimpleNamespace(can_transcribe=False, can_speak=False)
        with mock.patch.object(entry, "build_voice", mock.Mock(return_value=fake)):
            result = self.run_main(["--voice"], server=FakeServer())
        self.assertIn("requirements_voice.txt", result.out)

    def test_max_turns_below_one_is_refused(self):
        result = self.run_main(["--max-turns", "0"])
        self.assertIsNotNone(result.error)
        result.build_app.assert_not_called()


class FrontendBuiltTests(unittest.TestCase):
    def test_needs_index_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(entry.frontend_built(Path(tmp)))
            (Path(tmp) / "index.html").write_text("x", encoding="utf-8")
            self.assertTrue(entry.frontend_built(Path(tmp)))


if __name__ == "__main__":
    unittest.main()
