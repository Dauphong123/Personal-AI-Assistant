import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from tests import stubs

stubs.install()

import tool_args
import tools
from graph import node


@tool
def save(path: str, content: str) -> str:
    """Save content."""
    return content


@tool
def store(items: list[str]) -> str:
    """Store items."""
    return ",".join(items)


class JsonStringTests(unittest.TestCase):
    """langchain_ollama parses any string argument that looks like JSON into a dict or list
    (`_parse_arguments_from_tool_call`), so a str parameter can receive one."""

    def test_dict_for_a_str_parameter_becomes_json_text(self):
        args, error = tool_args.prepare(save, {"path": "a.json", "content": {"debug": True}})
        self.assertIsNone(error)
        self.assertEqual(args["content"], '{\n  "debug": true\n}')

    def test_list_for_a_str_parameter_becomes_json_text(self):
        args, _ = tool_args.prepare(save, {"path": "a.json", "content": ["é", 1]})
        self.assertEqual(args["content"], '[\n  "é",\n  1\n]')

    def test_a_real_list_parameter_is_left_alone(self):
        args, _ = tool_args.prepare(store, {"items": ["a", "b"]})
        self.assertEqual(args["items"], ["a", "b"])

    def test_plain_strings_are_unchanged(self):
        args, _ = tool_args.prepare(save, {"path": "a.txt", "content": "hi"})
        self.assertEqual(args, {"path": "a.txt", "content": "hi"})

    def test_write_file_through_the_graph_writes_the_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            call = {"name": "write_file", "args": {"path": "a.json", "content": {"debug": True}}, "id": "1"}

            with patch.object(tools, "PROJECT_ROOT", root):
                output = node._run_tool(call)

            self.assertIn("Successfully wrote", output)
            self.assertEqual((root / "a.json").read_text(encoding="utf-8"), '{\n  "debug": true\n}')


class ArgumentNameTests(unittest.TestCase):
    """The tools name a directory `path`, `directory`, `repo_path` or `dir`; pydantic dropped an
    unknown argument silently, so git_status(path="memories") reported on "." instead."""

    def test_directory_synonym_is_renamed(self):
        args, error = tool_args.prepare(tools.git_status, {"path": "memories"})
        self.assertIsNone(error)
        self.assertEqual(args, {"dir": "memories"})

    def test_each_project_directory_name_reaches_its_tool(self):
        cases = [
            (tools.list_files, "directory", "path"),
            (tools.search_code, "path", "directory"),
            (tools.git_diff, "dir", "repo_path"),
            (tools.git_log, "repo_path", "dir"),
        ]
        for tool, given, expected in cases:
            with self.subTest(tool=tool.name, given=given):
                args, error = tool_args.prepare(tool, {given: "x"})
                self.assertIsNone(error)
                self.assertEqual(args, {expected: "x"})

    def test_common_names_from_other_agents(self):
        args, error = tool_args.prepare(
            tools.edit_file, {"file_path": "a.py", "old_string": "x", "new_string": "y"}
        )
        self.assertIsNone(error)
        self.assertEqual(args, {"path": "a.py", "old": "x", "new": "y"})

        args, _ = tool_args.prepare(tools.terminal, {"cmd": "dir"})
        self.assertEqual(args, {"command": "dir"})

        args, _ = tool_args.prepare(tools.move_file, {"src": "a", "dest": "b"})
        self.assertEqual(args, {"source": "a", "destination": "b"})

    def test_a_synonym_is_not_used_when_the_real_name_is_given_too(self):
        _, error = tool_args.prepare(tools.read_file, {"path": "a.py", "file": "b.py"})
        self.assertIsNotNone(error)
        assert error is not None
        self.assertIn("unknown argument 'file'", error)

    def test_unknown_argument_is_refused_with_the_signature(self):
        _, error = tool_args.prepare(tools.read_file, {"path": "a.py", "lines": "1-5"})
        self.assertEqual(
            error,
            "ERROR: read_file got unknown argument 'lines'. Expected: "
            "read_file(path: str, start_line: int | None = None, end_line: int | None = None)",
        )

    def test_unknown_argument_never_reaches_the_tool(self):
        call = {"name": "git_status", "args": {"dir": ".", "verbose": True}, "id": "1"}
        with patch.object(tools, "_git", side_effect=AssertionError("ran")):
            output = node._run_tool(call)
        self.assertIn("unknown argument 'verbose'", output)


class ApprovalTests(unittest.TestCase):
    def run_node(self, args, answer=True):
        call = AIMessage(content="", tool_calls=[{"name": "write_file", "args": args, "id": "1"}])
        state: Any = {"messages": [HumanMessage(content="go", id="h"), call]}
        asked = []

        def interrupt(payload):
            asked.append(payload)
            return answer

        with tempfile.TemporaryDirectory() as tmp, patch.object(tools, "PROJECT_ROOT", Path(tmp).resolve()):
            with patch.object(node, "interrupt", interrupt):
                [message] = node.tool_execute(state)["messages"]

        return asked, message.content

    def test_the_user_approves_the_repaired_arguments(self):
        asked, _ = self.run_node({"file_path": "a.json", "content": {"a": 1}})
        self.assertEqual(
            asked, [{"calls": [{"name": "write_file", "args": {"path": "a.json", "content": '{\n  "a": 1\n}'}}]}]
        )

    def test_a_call_that_will_be_refused_is_not_put_to_the_user(self):
        asked, output = self.run_node({"path": "a.txt", "content": "x", "mode": "append"})
        self.assertEqual(asked, [])
        self.assertIn("unknown argument 'mode'", output)


if __name__ == "__main__":
    unittest.main()
