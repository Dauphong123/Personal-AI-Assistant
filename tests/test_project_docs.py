"""The project bookkeeping stays consistent: feature-lists.json (the feature log of the
new-feature workflow) and the Active tasks in docs/ROSTER.md (the session hand-off)."""

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FEATURES_FILE = ROOT / "feature-lists.json"
ROSTER_FILE = ROOT / "docs" / "ROSTER.md"

FEATURE_STATUSES = {"open", "planned", "in_progress", "done", "blocked"}
STEP_STATUSES = {"todo", "in_progress", "done", "blocked"}
TASK_STATUSES = {"todo", "doing", "blocked", "done"}


def load_features() -> list[dict]:
    return json.loads(FEATURES_FILE.read_text(encoding="utf-8"))["features"]


def roster_tasks() -> list[str]:
    """Lines of the 'Active tasks' section that are tasks (not the format line or '(none yet)')."""
    text = ROSTER_FILE.read_text(encoding="utf-8")
    section = text.split("## Active tasks", 1)[1].split("\n## ", 1)[0]
    return [line for line in section.splitlines() if re.match(r"^- \[\w+\]", line)]


@unittest.skipUnless(FEATURES_FILE.is_file(), "feature-lists.json not present")
class FeatureListTests(unittest.TestCase):
    def test_ids_are_unique_and_sequential(self):
        ids = [f["id"] for f in load_features()]
        self.assertEqual(ids, [f"F{n:03d}" for n in range(1, len(ids) + 1)])

    def test_statuses_are_known(self):
        for feature in load_features():
            self.assertIn(feature["status"], FEATURE_STATUSES, feature["id"])
            for step in feature.get("steps", []):
                self.assertIn(step["status"], STEP_STATUSES, f"{feature['id']} step {step.get('n')}")

    def test_done_features_are_finished(self):
        for feature in load_features():
            if feature["status"] != "done":
                continue
            self.assertTrue(feature.get("completed"), f"{feature['id']} has no completed date")
            unfinished = [s.get("n") for s in feature.get("steps", []) if s["status"] != "done"]
            self.assertEqual(unfinished, [], f"{feature['id']} is done but has unfinished steps")

    def test_branch_names_follow_the_workflow(self):
        for feature in load_features():
            branch = feature.get("branch")
            if branch:
                self.assertRegex(branch, rf"^feature/{feature['id']}-[a-z0-9-]+$", feature["id"])


@unittest.skipUnless(FEATURES_FILE.is_file() and ROSTER_FILE.is_file(), "bookkeeping files not present")
class RosterTests(unittest.TestCase):
    def test_task_lines_use_the_format(self):
        for line in roster_tasks():
            match = re.match(r"^- \[(\w+)\]", line)
            assert match is not None
            status = match.group(1)
            self.assertIn(status, TASK_STATUSES, line)

    def test_tasks_name_only_features_that_exist(self):
        known = {f["id"] for f in load_features()}
        for line in roster_tasks():
            for feature_id in re.findall(r"\bF\d{3}\b", line):
                self.assertIn(feature_id, known, f"roster names unknown {feature_id}: {line}")

    def test_finished_features_are_not_open_tasks(self):
        # ROSTER rule: remove a task once it is done and logged in PROJECT.md.
        status = {f["id"]: f["status"] for f in load_features()}
        for line in roster_tasks():
            for feature_id in re.findall(r"\bF\d{3}\b", line):
                self.assertNotEqual(status.get(feature_id), "done", f"stale roster task: {line}")

    def test_open_section_says_none_when_empty(self):
        text = ROSTER_FILE.read_text(encoding="utf-8")
        section = text.split("## Active tasks", 1)[1].split("\n## ", 1)[0]
        if not roster_tasks():
            self.assertIn("(none yet)", section)


class WebChatDocsTests(unittest.TestCase):
    """The web chat is two parts (agent API and frontend server); the docs say how to start both."""

    def test_commands_start_both_parts(self):
        for name in ("CLAUDE.md", "README.md"):
            text = (ROOT / name).read_text(encoding="utf-8")

            for command in ("python -m backend", "npm --prefix frontend start", "--serve-frontend"):
                self.assertIn(command, text, f"{name}: {command}")

if __name__ == "__main__":
    unittest.main()
