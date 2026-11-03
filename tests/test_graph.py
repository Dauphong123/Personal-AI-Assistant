import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.types import Command

from tests import stubs

stubs.install()

import tools  # noqa: E402
from graph import ModelGraph, node  # noqa: E402


def tool_call(name, args, id="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id}])


class ToolExecuteTests(unittest.TestCase):
    def run_node(self, calls):
        state = {"messages": [HumanMessage(content="go", id="h"), calls]}
        return node.tool_execute(state)["messages"]  # type: ignore[arg-type]

    def test_unknown_tool_becomes_error_message_not_crash(self):
        [msg] = self.run_node(tool_call("nope", {}))
        self.assertIsInstance(msg, ToolMessage)
        self.assertIn("unknown tool 'nope'", msg.content)

    def test_tool_exception_becomes_error_message(self):
        [msg] = self.run_node(tool_call("read_file", {"path": "../../etc/passwd"}))
        self.assertIn("ERROR: PathOutsideProjectError", msg.content)

    def test_knowledge_tool_is_dispatched(self):
        [msg] = self.run_node(tool_call("search_document", {"query": "x"}))
        self.assertEqual(msg.content, "doc")

    def test_tool_call_limit(self):
        done = [
            ToolMessage(content="x", tool_call_id=str(i))
            for i in range(node.MAX_TOOL_CALLS_PER_TURN)
        ]
        state = {
            "messages": [
                HumanMessage(content="go", id="h"),
                *done,
                tool_call("terminal", {"command": "x"}),
            ]
        }
        [msg] = node.tool_execute(state)["messages"]  # type: ignore[arg-type]
        self.assertIn("limit reached", msg.content)


class ApprovalFlowTests(unittest.TestCase):
    def run_graph(self, resume):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name).resolve()
        patches = [
            patch.object(tools, "PROJECT_ROOT", root),
            patch.object(
                node,
                "router_chain",
                RunnableLambda(lambda x: SimpleNamespace(destination="code")),
            ),
        ]
        replies = iter(
            [
                tool_call("write_file", {"path": "out.txt", "content": "hi"}),
                AIMessage(content="done"),
            ]
        )
        patches.append(patch.object(node, "code_chain", RunnableLambda(lambda x: next(replies))))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        app = ModelGraph.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "t"}}
        first = app.invoke(
            {"messages": [HumanMessage(content="write it")], "search_file": []}, config  # type: ignore[arg-type]
        )
        self.assertIn("__interrupt__", first)
        self.assertFalse((root / "out.txt").exists(), "nothing may run before approval")

        final = app.invoke(Command(resume=resume), config)  # type: ignore[arg-type]
        return root, final

    def test_approved_call_runs(self):
        root, final = self.run_graph(True)
        self.assertEqual((root / "out.txt").read_text(), "hi")
        self.assertEqual(final["messages"][-1].content, "done")

    def test_declined_call_does_not_run(self):
        root, final = self.run_graph(False)
        self.assertFalse((root / "out.txt").exists())
        tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
        self.assertIn("declined", tool_messages[0].content)


class SummarizeTests(unittest.TestCase):
    def history(self):
        return add_messages(
            [],
            [
                HumanMessage(content="q1"),
                AIMessage(content="a1"),
                HumanMessage(content="q2"),
                tool_call("read_file", {"path": "x"}, "c9"),
                ToolMessage(content="file body", tool_call_id="c9"),
                AIMessage(content="a2"),
            ],
        )

    def test_old_turns_summarized_and_last_turn_kept_intact_and_in_order(self):
        seen = {}
        chain = RunnableLambda(lambda x: seen.update(x) or "SUMMARY")
        messages = self.history()
        with patch.object(node, "summerize_chain", chain):
            update = node.summarize({"messages": messages})  # type: ignore[arg-type]

        result = add_messages(messages, update["messages"])
        self.assertEqual(
            [m.content for m in result][:2],
            ["Summary of the earlier conversation:\nSUMMARY", "q2"],
        )
        self.assertEqual([m.type for m in result], ["ai", "human", "ai", "tool", "ai"])
        self.assertEqual(result[3].tool_call_id, result[2].tool_calls[0]["id"])
        self.assertIn("q1", seen["messages"])
        self.assertNotIn("q2", seen["messages"])

    def test_single_turn_is_left_alone(self):
        messages = add_messages([], [HumanMessage(content="q"), AIMessage(content="a")])
        self.assertEqual(node.summarize({"messages": messages}), {})  # type: ignore[arg-type]


class RouterTests(unittest.TestCase):
    def test_router_passes_recent_history(self):
        seen = {}
        chain = RunnableLambda(
            lambda x: seen.update(x) or SimpleNamespace(destination="code")
        )
        messages = add_messages(
            [],
            [
                HumanMessage(content="fix the bug in main.py"),
                AIMessage(content="want me to?"),
                HumanMessage(content="yes"),
            ],
        )
        with patch.object(node, "router_chain", chain):
            self.assertEqual(node.router({"messages": messages}), "code_tools")  # type: ignore[arg-type]
        self.assertEqual(seen["question"], "yes")
        self.assertIn("fix the bug in main.py", seen["history"])


if __name__ == "__main__":
    unittest.main()
