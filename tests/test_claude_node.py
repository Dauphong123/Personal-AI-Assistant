"""F036: a turn whose model is Claude Code goes to the `claude_code` node (no router), streams its text
like any model, reports its tools, keeps one Claude session per thread and recaps talk it missed.
No real `claude` runs: ClaudeCodeChat gets a fake runner."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from tests import stubs

stubs.install()

import claude_bridge  # noqa: E402
from claude_bridge import ClaudeCodeError, Done, Text, ToolResult, ToolStart  # noqa: E402
from graph import ModelGraph, node  # noqa: E402
from model_choice import ModelChoice  # noqa: E402
from turns import TurnCancelled, run_turn  # noqa: E402

CLAUDE = ModelChoice("claude-code:sonnet", "high")
LOCAL = ModelChoice("qwen3:8b", None)


def no_router(_):
    raise AssertionError("the router must not run for Claude Code")


class FakeClaude:
    """Stands in for run_claude; each turn takes the next script."""

    def __init__(self):
        self.scripts: list = []
        self.runs: list[dict] = []

    def __call__(self, command, prompt, cwd, approve, cancel):
        self.runs.append({"command": command, "prompt": prompt, "cwd": cwd, "approve": approve})
        script = self.scripts.pop(0)
        if isinstance(script, BaseException):
            raise script
        yield from script

    def flag(self, run: int, name: str) -> str | None:
        command = self.runs[run]["command"]
        return command[command.index(name) + 1] if name in command else None


class ClaudeNodeTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeClaude()
        real = claude_bridge.ClaudeCodeChat
        fake = self.fake

        def chat(**fields):
            return real(runner=fake, **fields)

        patches = [
            patch.dict("os.environ", {"MODEL": "qwen3:8b"}),
            patch.object(node, "router_chain", RunnableLambda(no_router)),
            patch.object(node, "ClaudeCodeChat", chat),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.app = ModelGraph.compile(checkpointer=InMemorySaver())

    def turn(self, text: str, model=CLAUDE, thread: str = "t", on_tool=None, approve=lambda r: True) -> str:
        out: list[str] = []
        run_turn(self.app, text, {"configurable": {"thread_id": thread}}, out.append, approve, model=model, on_tool=on_tool)
        return "".join(out)

    def messages(self, thread: str = "t") -> list:
        return self.app.get_state({"configurable": {"thread_id": thread}}).values["messages"]  # type: ignore[arg-type]

    def test_claude_turn_skips_the_router_and_streams(self):
        self.fake.scripts.append([Text("Hel"), Text("lo"), Done("Hello", False)])

        self.assertEqual(self.turn("hi there friend"), "Hello")

        last = self.messages()[-1]
        self.assertIsInstance(last, AIMessage)
        self.assertEqual(last.content, "Hello")
        self.assertEqual(last.additional_kwargs["model"], "claude-code:sonnet")
        self.assertEqual(self.fake.flag(0, "--model"), "sonnet")
        self.assertEqual(self.fake.flag(0, "--effort"), "high")
        self.assertEqual(self.fake.runs[0]["cwd"], str(node.PROJECT_ROOT))

    def test_fallback_model_comes_from_the_setting(self):
        # F038 step 3
        self.fake.scripts += [[Done("a", False)], [Done("b", False)]]
        with patch.dict("os.environ", {"CLAUDE_FALLBACK_MODEL": "haiku"}):
            self.turn("first question here")
        with patch.dict("os.environ", {"CLAUDE_FALLBACK_MODEL": ""}):
            self.turn("second question here")

        self.assertEqual(self.fake.flag(0, "--fallback-model"), "haiku")
        self.assertIsNone(self.fake.flag(1, "--fallback-model"))

    def test_one_claude_session_per_thread(self):
        self.fake.scripts += [[Text("a"), Done("a", False)], [Text("b"), Done("b", False)], [Text("c"), Done("c", False)]]
        self.turn("first question here")
        self.turn("second question here")
        self.turn("other thread question", thread="u")

        self.assertIsNotNone(self.fake.flag(0, "--session-id"))
        self.assertEqual(self.fake.flag(1, "--resume"), self.fake.flag(0, "--session-id"))
        self.assertIsNotNone(self.fake.flag(2, "--session-id"))
        self.assertNotEqual(self.fake.flag(2, "--session-id"), self.fake.flag(0, "--session-id"))

    def test_error_becomes_the_answer_and_the_next_turn_works(self):
        self.fake.scripts += [ClaudeCodeError("not logged in"), [Text("ok"), Done("ok", False)]]

        self.assertEqual(self.turn("hello there friend"), "Claude Code could not answer: not logged in")
        self.assertEqual(self.turn("again please now"), "ok")
        # the failed turn started no session, so the next one starts it
        self.assertIsNotNone(self.fake.flag(1, "--session-id"))

    def test_a_failed_first_turn_never_reuses_its_session_id(self):
        # Claude Code may have created that session before failing; the same id again would clash.
        self.fake.scripts += [ClaudeCodeError("stopped"), [Done("", False)]]
        self.turn("hello there friend")
        self.turn("again please now")
        self.assertIsNotNone(self.fake.flag(1, "--session-id"))
        self.assertNotEqual(self.fake.flag(1, "--session-id"), self.fake.flag(0, "--session-id"))

    def test_a_restart_with_the_same_thread_id_starts_a_new_claude_session(self):
        # The CLI always uses thread "1"; after a restart the old Claude session id exists already.
        self.fake.scripts += [[Done("", False)], [Done("", False)]]
        self.turn("first run question")
        self.app = ModelGraph.compile(checkpointer=InMemorySaver())  # a new program run
        self.turn("second run question")
        self.assertIsNone(self.fake.flag(1, "--resume"))
        self.assertNotEqual(self.fake.flag(1, "--session-id"), self.fake.flag(0, "--session-id"))

    def test_two_tool_calls_with_clashing_arguments_do_not_break_the_turn(self):
        self.fake.scripts.append(
            [
                ToolStart("t1", "Edit", {"file_path": "a.py", "replace_all": False}),
                ToolResult("t1", "ok", False),
                ToolStart("t2", "Edit", {"file_path": "b.py", "replace_all": True}),
                ToolResult("t2", "ok", False),
                Text("Edited both."),
                Done("Edited both.", False),
            ]
        )
        reports: list[dict] = []
        self.assertEqual(self.turn("edit both files", on_tool=reports.append), "Edited both.")
        self.assertEqual([(r["id"], r["status"]) for r in reports], [("t1", "running"), ("t1", "done"), ("t2", "running"), ("t2", "done")])

    def test_cancel_propagates_and_the_next_turn_works(self):
        self.fake.scripts += [TurnCancelled(), [Text("ok"), Done("ok", False)]]

        with self.assertRaises(TurnCancelled):
            self.turn("long task please now")

        self.assertEqual(self.turn("next one please"), "ok")

    def test_tools_are_reported_apart_from_the_text(self):
        self.fake.scripts.append(
            [
                Text("Reading."),
                ToolStart("t1", "Read", {"file_path": "a.py"}),
                ToolResult("t1", "print(1)", False),
                Text(" Done."),
                Done("Reading. Done.", False),
            ]
        )
        reports: list[dict] = []

        text = self.turn("read a.py please", on_tool=reports.append)

        self.assertEqual(text, "Reading. Done.")
        self.assertEqual([r["status"] for r in reports], ["running", "done"])
        self.assertEqual(reports[1]["text"], "print(1)")
        self.assertEqual(reports[0]["name"], "Read")

    def test_approve_reaches_claude(self):
        asked: list[dict] = []

        def approve(request: dict) -> bool:
            asked.append(request)
            return False

        self.fake.scripts.append([Done("", False)])
        self.turn("write a file please", approve=approve)
        self.assertFalse(self.fake.runs[0]["approve"]({"calls": []}))
        self.assertEqual(asked, [{"calls": []}])

    def test_local_model_still_uses_the_router(self):
        with patch.object(node, "router_chain", RunnableLambda(lambda x: SimpleNamespace(destination="direct"))), \
             patch.object(node, "direct_chain", RunnableLambda(lambda x: AIMessage(content="local"))):
            self.turn("hello there friend", model=LOCAL)
        self.assertEqual(self.fake.runs, [])
        self.assertEqual(self.messages()[-1].content, "local")


class ClaudePromptTests(unittest.TestCase):
    def test_first_claude_turn_recaps_earlier_talk_after_memory(self):
        messages = [
            HumanMessage(content="What is Python?"),
            AIMessage(content="A language."),
            HumanMessage(content="Read setup.py"),
            AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "c"}]),
            ToolMessage(content="secret file text", tool_call_id="c"),
            AIMessage(content="It installs the app."),
            HumanMessage(content="Now improve it"),
        ]
        prompt = node.claude_prompt(messages, "The user is Dat.", resumed=False)

        self.assertLess(prompt.index("The user is Dat."), prompt.index("What is Python?"))
        self.assertIn("A language.", prompt)
        self.assertIn("It installs the app.", prompt)
        self.assertNotIn("secret file text", prompt)
        self.assertTrue(prompt.rstrip().endswith("Now improve it"))
        self.assertEqual(prompt.count("Now improve it"), 1)

    def test_resumed_turn_with_nothing_new_is_just_the_message(self):
        messages = [
            HumanMessage(content="hi"),
            AIMessage(content="hello", additional_kwargs={"model": "claude-code:opus"}),
            HumanMessage(content="next"),
        ]
        self.assertEqual(node.claude_prompt(messages, "", resumed=True), "next")

    def test_only_talk_after_the_last_claude_answer_is_recapped(self):
        messages = [
            HumanMessage(content="old question"),
            AIMessage(content="claude answer", additional_kwargs={"model": "claude-code:opus"}),
            HumanMessage(content="local question"),
            AIMessage(content="local answer"),
            HumanMessage(content="back to claude"),
        ]
        prompt = node.claude_prompt(messages, "", resumed=True)
        self.assertIn("local question", prompt)
        self.assertIn("local answer", prompt)
        self.assertNotIn("old question", prompt)

    def test_notices_are_left_out_and_long_messages_cut(self):
        messages = [
            HumanMessage(content="[system notice] job 3 finished", additional_kwargs={"notice": True}),
            AIMessage(content="x" * 2000),
            HumanMessage(content="go"),
        ]
        prompt = node.claude_prompt(messages, "", resumed=False)
        self.assertNotIn("job 3", prompt)
        self.assertLess(prompt.count("x"), 600)


if __name__ == "__main__":
    unittest.main()
