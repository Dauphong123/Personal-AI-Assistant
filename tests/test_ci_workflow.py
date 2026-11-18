"""The GitHub Actions workflow (.github/workflows/ci.yml) checks every push the way the
commands in CLAUDE.md do: Python unit tests in a .venv (where test_typecheck finds pyright)
and the TypeScript frontend's type check, tests and build."""

import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def load_workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def triggers(workflow: dict) -> dict:
    # PyYAML (YAML 1.1) reads the bare key `on` as the boolean True.
    on = workflow.get("on", workflow.get(True))
    assert isinstance(on, dict), "workflow has no `on:` triggers"
    return on


def run_lines(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def uses(job: dict) -> list[str]:
    return [step["uses"] for step in job["steps"] if "uses" in step]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(WORKFLOW.is_file(), "missing .github/workflows/ci.yml")
        self.workflow = load_workflow()
        self.jobs = self.workflow["jobs"]

    def test_runs_on_every_push_and_on_pull_requests_to_main(self):
        on = triggers(self.workflow)
        self.assertIn("push", on)
        push = on["push"] or {}
        self.assertNotIn("branches", push, "push must not be limited to some branches")
        self.assertEqual(on["pull_request"]["branches"], ["main"])

    def test_newer_push_cancels_the_older_run(self):
        self.assertTrue(self.workflow["concurrency"]["cancel-in-progress"])

    def test_python_job_matches_the_local_setup(self):
        job = self.jobs["python"]
        self.assertEqual(job["runs-on"], "windows-latest")  # ctypes WinDLL, .venv\Scripts paths
        setup = next(s for s in job["steps"] if s.get("uses", "").startswith("actions/setup-python"))
        self.assertEqual(str(setup["with"]["python-version"]), "3.12")
        script = run_lines(job)
        self.assertIn("python -m venv .venv", script)
        self.assertIn("-r requirements.txt", script)
        self.assertIn("-r requirements_voice.txt", script)  # pyright resolves the voice imports
        self.assertRegex(script, r"pyright==\d")
        self.assertIn("python -m unittest discover -s tests -t .", script)

    def test_python_job_builds_the_frontend_its_tests_check(self):
        script = run_lines(self.jobs["python"])
        self.assertIn("npm --prefix webui ci", script)
        self.assertIn("npm --prefix webui run build", script)

    def test_python_job_installs_ripgrep(self):
        self.assertIn("ripgrep", run_lines(self.jobs["python"]))

    def test_webui_job_checks_tests_and_builds(self):
        job = self.jobs["webui"]
        self.assertTrue(any(u.startswith("actions/setup-node") for u in uses(job)))
        script = run_lines(job)
        for command in ("npm ci", "npm run check", "npm test", "npm run build"):
            self.assertIn(command, script)
        self.assertEqual(job["defaults"]["run"]["working-directory"], "webui")

    def test_actions_are_pinned_to_a_major_version(self):
        for name, job in self.jobs.items():
            for action in uses(job):
                self.assertRegex(action, r"^[\w-]+/[\w-]+@v\d+$", f"{name}: {action}")


if __name__ == "__main__":
    unittest.main()
