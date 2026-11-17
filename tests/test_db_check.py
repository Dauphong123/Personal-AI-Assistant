import contextlib
import io
import subprocess
import unittest
from unittest import mock

from memories import check
from memories.diagnose import BAD_CREDENTIALS, OK, UNREACHABLE, WRONG_SERVER

URL = "postgresql+psycopg://agent:s3cret@localhost:5432/agent_memory"
AUTH_ERROR = 'FATAL:  password authentication failed for user "agent"'

NETSTAT = "  TCP    0.0.0.0:5432           0.0.0.0:0              LISTENING       6688\n"
TASKLIST = '"postgres.exe","6688","Services","0","20,000 K"\n'


def fake_run(netstat=NETSTAT, tasklist=TASKLIST):
    def run(cmd, **kwargs):
        text = netstat if cmd[0] == "netstat" else tasklist
        return subprocess.CompletedProcess(cmd, 0, stdout=text, stderr="")

    return run


class PortOwnersTests(unittest.TestCase):
    def test_resolves_pid_to_process_name(self):
        self.assertEqual(check.port_owners(5432, run=fake_run()), ["postgres.exe"])

    def test_no_listener_gives_empty_list(self):
        self.assertEqual(check.port_owners(5432, run=fake_run(netstat="")), [])

    def test_missing_tools_give_empty_list(self):
        def broken(cmd, **kwargs):
            raise FileNotFoundError(cmd[0])

        self.assertEqual(check.port_owners(5432, run=broken), [])


class RunCheckTests(unittest.TestCase):
    def test_ok(self):
        status, report = check.run_check(URL, connect=lambda url: None, owners=lambda port: [])
        self.assertEqual(status, OK)
        self.assertNotIn("s3cret", report)

    def test_wrong_server_names_the_process_and_the_fix(self):
        status, report = check.run_check(
            URL, connect=lambda url: AUTH_ERROR, owners=lambda port: ["postgres.exe"]
        )
        self.assertEqual(status, WRONG_SERVER)
        self.assertIn("postgres.exe", report)
        self.assertIn("5432", report)
        self.assertIn("15432", report)  # points at the free host port in compose.yaml
        self.assertNotIn("s3cret", report)

    def test_wrong_server_beside_docker_names_only_the_foreign_process(self):
        status, report = check.run_check(
            URL,
            connect=lambda url: AUTH_ERROR,
            owners=lambda port: ["com.docker.backend.exe", "postgres.exe"],
        )
        self.assertEqual(status, WRONG_SERVER)
        self.assertIn("answered by postgres.exe", report)

    def test_wrong_server_on_the_suggested_port_says_pick_another(self):
        url = URL.replace(":5432/", f":{check.SUGGESTED_HOST_PORT}/")
        _, report = check.run_check(
            url, connect=lambda url: AUTH_ERROR, owners=lambda port: ["postgres.exe"]
        )
        self.assertIn("free port", report)

    def test_bad_credentials(self):
        status, report = check.run_check(
            URL, connect=lambda url: AUTH_ERROR, owners=lambda port: ["com.docker.backend.exe"]
        )
        self.assertEqual(status, BAD_CREDENTIALS)
        self.assertIn("POSTGRES_PASSWORD", report)

    def test_unreachable_suggests_starting_the_container(self):
        status, report = check.run_check(
            URL, connect=lambda url: "connection refused", owners=lambda port: []
        )
        self.assertEqual(status, UNREACHABLE)
        self.assertIn("docker compose up -d", report)


class ExplainFailureTests(unittest.TestCase):
    def test_names_the_process_from_the_error_already_caught(self):
        with mock.patch.object(check, "get_database_url", return_value=URL):
            hint = check.explain_failure(
                AUTH_ERROR, owners=lambda port: ["postgres.exe"]
            )
        self.assertIn("postgres.exe", hint)
        self.assertIn("python -m memories.check", hint)
        self.assertNotIn("s3cret", hint)

    def test_never_raises_when_url_is_missing(self):
        with mock.patch.object(check, "get_database_url", side_effect=RuntimeError("x")):
            hint = check.explain_failure(AUTH_ERROR)
        self.assertIn("python -m memories.check", hint)


class MainTests(unittest.TestCase):
    def run_main(self, status):
        out = io.StringIO()
        with (
            mock.patch.object(check, "get_database_url", return_value=URL),
            mock.patch.object(check, "run_check", return_value=(status, "report text")),
            contextlib.redirect_stdout(out),
        ):
            code = check.main([])
        return code, out.getvalue()

    def test_exit_code_zero_only_when_ok(self):
        self.assertEqual(self.run_main(OK)[0], 0)
        self.assertEqual(self.run_main(WRONG_SERVER)[0], 1)

    def test_prints_the_report(self):
        self.assertIn("report text", self.run_main(OK)[1])

    def test_missing_url_exits_nonzero_with_message(self):
        out = io.StringIO()
        with (
            mock.patch.object(check, "get_database_url", side_effect=RuntimeError("Set DATABASE_URL")),
            contextlib.redirect_stdout(out),
        ):
            code = check.main([])
        self.assertEqual(code, 1)
        self.assertIn("DATABASE_URL", out.getvalue())


if __name__ == "__main__":
    unittest.main()
