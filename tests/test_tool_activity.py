"""Tool activity: a UI can show tool calls apart from the answer (run_turn on_tool)."""

import unittest
from types import SimpleNamespace

from langchain_core.messages import AIMessageChunk, ToolMessage

from tests import stubs

stubs.install()

import turns


def call_chunk(name, args, id="c1", message="a"):
    return AIMessageChunk(content="", id=message, tool_calls=[{"name": name, "args": args, "id": id}])


def result(text, id="c1"):
    return ToolMessage(content=text, tool_call_id=id)


class PhasedApp:
    """A compiled graph stand-in: each stream() call yields the next phase's (node, chunk)
    pairs; every phase but the last ends paused on an approval."""

    def __init__(self, *phases):
        self.phases = list(phases)
        self.streams = 0

    def stream(self, graph_input, config=None, stream_mode=None):
        pairs = self.phases[self.streams]
        self.streams += 1
        for node_name, piece in pairs:
            yield piece, {"langgraph_node": node_name}

    def get_state(self, config):
        waiting = self.streams < len(self.phases)
        tasks = [SimpleNamespace(interrupts=[SimpleNamespace(value={"calls": []})])] if waiting else []
        return SimpleNamespace(tasks=tasks, values={"messages": []})


def run(*phases, on_tool=True):
    text, tools = [], []
    turns.run_turn(PhasedApp(*phases), "q", {}, text.append, lambda request: True, on_tool=tools.append if on_tool else None)
    return "".join(text), tools


class ToolActivityTests(unittest.TestCase):
    def test_tool_output_goes_to_on_tool_not_into_the_answer(self):
        text, tools = run(
            [
                ("code_tools", AIMessageChunk(content="Let me look. ", id="a")),
                ("code_tools", call_chunk("read_file", {"path": "main.py"})),
                ("tool_execute", result("FILE CONTENTS")),
                ("code_tools", AIMessageChunk(content="It starts the chat.", id="b")),
            ]
        )
        self.assertEqual(text, "Let me look. It starts the chat.")
        self.assertEqual(
            tools,
            [
                {"id": "c1", "name": "read_file", "args": {"path": "main.py"}, "status": "running"},
                {"id": "c1", "name": "read_file", "args": {"path": "main.py"}, "status": "done", "text": "FILE CONTENTS"},
            ],
        )

    def test_without_on_tool_tool_output_still_streams_into_the_text(self):
        text, _ = run([("code_tools", call_chunk("read_file", {"path": "x"})), ("tool_execute", result("RAW"))], on_tool=False)
        self.assertEqual(text, "RAW")

    def test_an_error_result_is_reported_as_an_error(self):
        _, tools = run([("code_tools", call_chunk("terminal", {"command": "x"})), ("tool_execute", result("ERROR: the user declined this action."))])
        self.assertEqual(tools[-1]["status"], "error")
        self.assertEqual(tools[-1]["text"], "ERROR: the user declined this action.")

    def test_a_call_split_across_chunks_is_announced_once_with_its_whole_arguments(self):
        pieces = [
            AIMessageChunk(content="", id="a", tool_call_chunks=[{"name": "search_code", "args": '{"pattern": ', "id": "c1", "index": 0}]),
            AIMessageChunk(content="", id="a", tool_call_chunks=[{"name": None, "args": '"route"}', "id": None, "index": 0}]),
        ]
        _, tools = run([*[("code_tools", p) for p in pieces], ("tool_execute", result("hits"))])
        self.assertEqual([t["status"] for t in tools], ["running", "done"])
        self.assertEqual(tools[0]["name"], "search_code")
        self.assertEqual(tools[-1]["args"], {"pattern": "route"})

    def test_a_result_after_an_approval_pause_keeps_the_call_name(self):
        # The model's call streams before the pause; its result arrives after the resume.
        _, tools = run(
            [("code_tools", call_chunk("terminal", {"command": "dir"}))],
            [("tool_execute", result("listing"))],
        )
        self.assertEqual(tools[-1], {"id": "c1", "name": "terminal", "args": {"command": "dir"}, "status": "done", "text": "listing"})

    def test_several_calls_in_one_message(self):
        message = AIMessageChunk(
            content="",
            id="a",
            tool_calls=[
                {"name": "read_file", "args": {"path": "a.py"}, "id": "c1"},
                {"name": "read_file", "args": {"path": "b.py"}, "id": "c2"},
            ],
        )
        _, tools = run([("code_tools", message), ("tool_execute", result("A", "c1")), ("tool_execute", result("B", "c2"))])
        self.assertEqual([(t["id"], t["status"]) for t in tools], [("c1", "running"), ("c2", "running"), ("c1", "done"), ("c2", "done")])

    def test_an_unknown_result_is_still_reported(self):
        _, tools = run([("tool_execute", result("orphan", "zz"))])
        self.assertEqual(tools, [{"id": "zz", "name": "tool", "args": {}, "status": "done", "text": "orphan"}])


if __name__ == "__main__":
    unittest.main()
