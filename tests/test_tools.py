import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import stubs

stubs.install()

import jobs  # noqa: E402
import tools  # noqa: E402


class ToolTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name).resolve()
        patcher = patch.object(tools, "PROJECT_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)
        # Temp dirs can sit inside another git repo; stop git searching upward.
        env = patch.dict(os.environ, {"GIT_CEILING_DIRECTORIES": str(self.root.parent)})
        env.start()
        self.addCleanup(env.stop)

    def call(self, name, **args):
        return tools.TOOL_BY_NAME[name].invoke(args)


class PathPolicyTests(ToolTestCase):
    def test_paths_outside_project_are_rejected(self):
        for bad in ["../outside.txt", str(self.root.parent / "x.txt")]:
            with self.assertRaises(tools.PathOutsideProjectError):
                self.call("read_file", path=bad)
            with self.assertRaises(tools.PathOutsideProjectError):
                self.call("write_file", path=bad, content="x")

    def test_git_directory_is_not_writable(self):
        (self.root / ".git").mkdir()
        with self.assertRaises(tools.PathOutsideProjectError):
            self.call("write_file", path=".git/config", content="x")

    def test_working_inside_project(self):
        self.call("write_file", path="a/b.txt", content="hi")
        self.assertEqual((self.root / "a/b.txt").read_text(), "hi")
        self.call("move_file", source="a/b.txt", destination="c.txt")
        self.assertTrue((self.root / "c.txt").exists())
        self.call("delete_file", path="c.txt")
        self.assertFalse((self.root / "c.txt").exists())


class ReadAndListTests(ToolTestCase):
    def test_read_file_returns_whole_small_file_by_default(self):
        (self.root / "f.txt").write_text("one\ntwo\nthree\n")
        self.assertEqual(self.call("read_file", path="f.txt"), "1: one\n2: two\n3: three")

    def test_read_file_range_and_truncation(self):
        (self.root / "big.txt").write_text("\n".join(f"l{i}" for i in range(1, 1001)))
        out = self.call("read_file", path="big.txt")
        self.assertIn("1: l1", out)
        self.assertIn("400: l400", out)
        self.assertNotIn("401: l401", out)
        self.assertIn("truncated", out)
        self.assertEqual(
            self.call("read_file", path="big.txt", start_line=10, end_line=11),
            "10: l10\n11: l11",
        )

    def test_list_files_skips_ignored_directories(self):
        (self.root / ".venv").mkdir()
        (self.root / ".venv" / "hidden.py").write_text("")
        (self.root / "keep.py").write_text("")
        self.assertEqual(self.call("list_files"), ["keep.py"])


class EditTests(ToolTestCase):
    def test_edit_requires_unique_match_or_occurrence(self):
        (self.root / "f.txt").write_text("a b a")
        self.assertIn("occurs 2 times", self.call("edit_file", path="f.txt", old="a", new="X"))
        self.call("edit_file", path="f.txt", old="a", new="X", occurrence=2)
        self.assertEqual((self.root / "f.txt").read_text(), "a b X")
        self.assertIn("must not be empty", self.call("edit_file", path="f.txt", old="", new="X"))


class CommandTests(ToolTestCase):
    def test_git_show_rejects_option_like_commit(self):
        self.assertIn("invalid commit", self.call("git_show", dir=".", commit="--output=evil"))

    def test_git_in_non_repository_is_an_error_string(self):
        self.assertTrue(self.call("git_log", dir=".", limit=-5).startswith("ERROR"))

    def test_credential_files_are_not_read(self):
        for name in [".env", ".env.local", ".env.production", "server.pem", "api.key", "id_rsa", "id_ed25519", ".netrc", ".pgpass", ".npmrc", "cert.p12"]:
            (self.root / name).write_text("SECRET_VALUE=hunter2\n")
            result = self.call("read_file", path=name)
            self.assertIn("may contain secrets", result, name)
            self.assertNotIn("hunter2", result, name)

    def test_templates_and_ordinary_files_are_still_read(self):
        for name in [".env.example", ".env.sample", "id_rsa.pub", "notes.txt", "keys.py", "environment.md"]:
            (self.root / name).write_text("harmless text\n")
            self.assertIn("harmless text", self.call("read_file", path=name), name)

    def test_the_name_check_ignores_case_and_folders(self):
        (self.root / "config").mkdir()
        (self.root / "config" / ".ENV").write_text("TOKEN=abc\n")
        self.assertIn("may contain secrets", self.call("read_file", path="config/.ENV"))

    @unittest.skipUnless(shutil.which("rg"), "ripgrep not installed")
    def test_search_code_does_not_reveal_credential_files(self):
        # ripgrep already skips hidden files such as .env; the visible key files need the filter.
        for name in [".env", "server.pem", "api.key", "id_rsa", "backup.p12"]:
            (self.root / name).write_text("DATABASE_PASSWORD=hunter2\n")

        (self.root / "app.py").write_text("password = load()  # hunter2 lives elsewhere\n")
        found = self.call("search_code", query="hunter2")
        self.assertIn("app.py:1:password = load()", found)

        for name in ["server.pem", "api.key", "id_rsa", "backup.p12", ".env"]:
            self.assertNotIn(name, found)

        self.assertNotIn("DATABASE_PASSWORD", found)

    @unittest.skipUnless(shutil.which("rg"), "ripgrep not installed")
    def test_search_code_says_so_when_only_credential_files_match(self):
        (self.root / "server.pem").write_text("hunter2\n")
        self.assertEqual(self.call("search_code", query="hunter2"), "No matches found for: hunter2")

    @unittest.skipUnless(shutil.which("rg"), "ripgrep not installed")
    def test_search_code_treats_dash_query_as_text(self):
        (self.root / "f.txt").write_text("a --version b\n")
        self.assertIn("--version", self.call("search_code", query="--version"))

    def python_job(self, code):
        return patch.object(jobs, "shell_args", return_value=[sys.executable, "-c", code])

    def test_terminal_returns_output_of_a_quick_command(self):
        with self.python_job("print('hi')"):
            self.assertEqual(self.call("terminal", command="x").strip(), "hi")

    def test_terminal_reports_failure(self):
        with self.python_job("import sys; sys.stderr.write('boom'); sys.exit(3)"):
            result = self.call("terminal", command="x")
        self.assertIn("exit code 3", result)
        self.assertIn("boom", result)

    def test_slow_terminal_command_goes_to_background_and_can_be_cancelled(self):
        with (
            self.python_job("import time; time.sleep(30)"),
            patch.object(tools, "FOREGROUND_WAIT", 0.2),
        ):
            result = self.call("terminal", command="x")

        job = tools.JOBS.all()[-1]
        self.addCleanup(tools.JOBS.cancel, job.id)
        self.assertIn(f"Job {job.id} is still running", result)
        self.assertIn("still running", self.call("job_status", job_id=job.id))
        self.assertIn("Cancelled", self.call("cancel_job", job_id=job.id))
        self.assertTrue(job.done.wait(10))
        self.assertIn("cancelled", self.call("job_status", job_id=job.id))

    def test_job_status_unknown_id_is_an_error_string(self):
        self.assertTrue(self.call("job_status", job_id=9999).startswith("ERROR"))


class RegistryTests(unittest.TestCase):
    def test_every_tool_is_dispatchable_and_approval_names_exist(self):
        names = [t.name for t in tools.TOOLS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), set(tools.TOOL_BY_NAME))
        self.assertIn("delete_file", names)
        self.assertIn("search_document", names)
        self.assertTrue(tools.NEEDS_APPROVAL <= set(names))


if __name__ == "__main__":
    unittest.main()
