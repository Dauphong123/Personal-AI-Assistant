import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
