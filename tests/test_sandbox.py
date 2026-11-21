"""F038 step 4: terminal jobs run boxed - no secret environment variables, a memory cap and,
on Windows, a job object that takes every process the command started down with it."""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import jobs
import sandbox
from jobs import JobManager
from settings import check_settings, terminal_memory_mb

WINDOWS = os.name == "nt"


class ChildEnvTests(unittest.TestCase):
    def test_secret_variables_are_dropped(self):
        env = {
            "PATH": "p",
            "PWD": "/x",
            "MONKEY": "m",
            "KEYBOARD_LAYOUT": "us",
            "GITHUB_TOKEN": "t",
            "OPENAI_API_KEY": "k",
            "AWS_SECRET_ACCESS_KEY": "s",
            "PGPASSWORD": "pw",
            "DB_PASSWORD": "pw",
            "DATABASE_URL": "postgresql://u:pw@h/db",
            "db_url": "postgresql://u:pw@h/db",
            "GH_AUTH": "a",
            "GOOGLE_APPLICATION_CREDENTIALS": "c.json",
            "ANTHROPIC_APIKEY": "k",
        }
        kept = sandbox.child_env(env)

        self.assertEqual(set(kept), {"PATH", "PWD", "MONKEY", "KEYBOARD_LAYOUT"})

    def test_the_real_environment_is_not_changed(self):
        with patch.dict(os.environ, {"F038_TEST_TOKEN": "t"}):
            sandbox.child_env()
            self.assertEqual(os.environ["F038_TEST_TOKEN"], "t")


class JobSandboxTests(unittest.TestCase):
    def setUp(self):
        self.cwd = Path(tempfile.gettempdir())
        self.manager = JobManager()
        self.addCleanup(self.manager.close)

    def run_code(self, code: str, **env) -> jobs.Job:
        with patch.object(jobs, "shell_args", return_value=[sys.executable, "-c", code]), patch.dict(os.environ, env):
            job = self.manager.start("cmd", cwd=self.cwd)
        self.assertTrue(self.manager.wait_or_detach(job, 30))
        return job

    def test_the_command_does_not_see_secrets(self):
        job = self.run_code(
            "import os; print(os.environ.get('F038_TEST_TOKEN'), os.environ.get('DB_URL'), bool(os.environ.get('PATH')))",
            F038_TEST_TOKEN="t",
            DB_URL="postgresql://agent:pw@localhost:15432/x",
        )
        self.assertEqual(job.report().strip(), "None None True")

    @unittest.skipUnless(WINDOWS, "job objects are Windows-only")
    def test_memory_over_the_cap_fails_the_command(self):
        with patch.dict(os.environ, {"TERMINAL_MEMORY_MB": "200"}):
            job = self.run_code("b = bytearray(600 * 1024 * 1024); print('allocated')")
        self.assertNotEqual(job.returncode, 0)
        self.assertNotIn("allocated", job.stdout)

    @unittest.skipUnless(WINDOWS, "job objects are Windows-only")
    def test_a_normal_command_fits_the_cap(self):
        with patch.dict(os.environ, {"TERMINAL_MEMORY_MB": "200"}):
            job = self.run_code("b = bytearray(20 * 1024 * 1024); print('ok')")
        self.assertEqual(job.returncode, 0)
        self.assertEqual(job.stdout.strip(), "ok")

    @unittest.skipUnless(WINDOWS, "job objects are Windows-only")
    def test_cancel_reaches_a_process_whose_parent_is_gone(self):
        # The parent starts a sleeping grandchild without pipes and exits; taskkill /T cannot find
        # it any more, the job object still can.
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        code = (
            "import subprocess, sys; "
            f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], creationflags={flags}, "
            "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); print(p.pid)"
        )
        job = self.run_code(code)
        pid = int(job.stdout.strip())
        self.assertTrue(sandbox.pid_alive(pid))

        self.manager.close()

        deadline = time.monotonic() + 10
        while sandbox.pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        self.assertFalse(sandbox.pid_alive(pid))


class SettingTests(unittest.TestCase):
    def test_default_and_values(self):
        self.assertEqual(terminal_memory_mb({}), 2048)
        self.assertEqual(terminal_memory_mb({"TERMINAL_MEMORY_MB": "0"}), 0)
        self.assertEqual(terminal_memory_mb({"TERMINAL_MEMORY_MB": "512"}), 512)

    def test_bad_values_are_listed_at_startup(self):
        for raw in ("-5", "lots", "50"):
            with self.subTest(raw=raw):
                problems = check_settings({"MODEL": "qwen3:8b", "TERMINAL_MEMORY_MB": raw})
                self.assertTrue(any("TERMINAL_MEMORY_MB" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
