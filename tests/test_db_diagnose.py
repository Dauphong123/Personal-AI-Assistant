import unittest

from memories.diagnose import (
    BAD_CREDENTIALS,
    OK,
    UNKNOWN,
    UNREACHABLE,
    WRONG_SERVER,
    classify,
    parse_netstat,
    redact_url,
)

NETSTAT = """
Active Connections

  Proto  Local Address          Foreign Address        State           PID
  TCP    0.0.0.0:5432           0.0.0.0:0              LISTENING       6688
  TCP    [::]:5432              [::]:0                 LISTENING       6688
  TCP    127.0.0.1:54320        0.0.0.0:0              LISTENING       111
  TCP    127.0.0.1:5432         127.0.0.1:50000        ESTABLISHED     6688
  TCP    0.0.0.0:15432          0.0.0.0:0              LISTENING       222
  UDP    0.0.0.0:5432           *:*                                    333
"""

AUTH_ERROR = (
    "connection failed: connection to server at \"127.0.0.1\", port 5432 failed: "
    'FATAL:  password authentication failed for user "agent"'
)


class ParseNetstatTests(unittest.TestCase):
    def test_returns_listening_pids_for_exactly_that_port(self):
        self.assertEqual(parse_netstat(NETSTAT, 5432), [6688])

    def test_other_port_and_empty_output(self):
        self.assertEqual(parse_netstat(NETSTAT, 15432), [222])
        self.assertEqual(parse_netstat("", 5432), [])


class ClassifyTests(unittest.TestCase):
    def test_no_error_is_ok(self):
        self.assertEqual(classify(None, []), OK)

    def test_auth_failure_with_foreign_listener_is_wrong_server(self):
        self.assertEqual(classify(AUTH_ERROR, ["postgres.exe"]), WRONG_SERVER)

    def test_auth_failure_with_docker_listener_is_bad_credentials(self):
        self.assertEqual(classify(AUTH_ERROR, ["com.docker.backend.exe"]), BAD_CREDENTIALS)
        self.assertEqual(classify(AUTH_ERROR, ["wslrelay.exe"]), BAD_CREDENTIALS)

    def test_auth_failure_with_docker_and_foreign_listener_is_wrong_server(self):
        # Windows lets a native postgres bind the same port as Docker; it may answer.
        owners = ["com.docker.backend.exe", "postgres.exe"]
        self.assertEqual(classify(AUTH_ERROR, owners), WRONG_SERVER)

    def test_auth_failure_with_unknown_listener_is_bad_credentials(self):
        self.assertEqual(classify(AUTH_ERROR, []), BAD_CREDENTIALS)

    def test_refused_or_timeout_is_unreachable(self):
        self.assertEqual(classify("connection refused", []), UNREACHABLE)
        self.assertEqual(classify("connection timeout expired", []), UNREACHABLE)

    def test_anything_else_is_unknown(self):
        self.assertEqual(classify('database "x" does not exist', ["postgres.exe"]), UNKNOWN)


class RedactTests(unittest.TestCase):
    def test_password_is_hidden(self):
        shown = redact_url("postgresql+psycopg://agent:s3cret@localhost:5432/agent_memory")
        self.assertNotIn("s3cret", shown)
        self.assertIn("agent", shown)
        self.assertIn("localhost:5432", shown)

    def test_url_without_password_is_unchanged(self):
        self.assertEqual(redact_url("postgresql://localhost/db"), "postgresql://localhost/db")


if __name__ == "__main__":
    unittest.main()
