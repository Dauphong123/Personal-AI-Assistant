import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from tests import stubs

stubs.install()

import chat_loop  # noqa: E402
import turns  # noqa: E402
from chat_loop import ChatLoop  # noqa: E402
from graph import ModelGraph, node  # noqa: E402
from jobs import JobManager  # noqa: E402
from turns import TurnCancelled, is_stop_request  # noqa: E402


def wait_for(condition, timeout=5):
    end = time.monotonic() + timeout

    while time.monotonic() < end:
        if condition():
            return True

        time.sleep(0.01)

    return False


class StopRequestTests(unittest.TestCase):
    def test_stop_phrases(self):
        for text in [
            "stop", "Stop!", "stop it", "please stop", "cancel that", "cancel",
            "abort", "never mind", "nevermind", "okay stop now", "/stop", "forget it",
        ]:  # fmt: skip
            self.assertTrue(is_stop_request(text), text)

    def test_requests_that_merely_mention_stop_are_not_stop(self):
        for text in [
            "stop the docker container",
            "how do I cancel a subscription",
            "don't stop",
            "explain the stop word list",
            "",
            "yes",
        ]:
            self.assertFalse(is_stop_request(text), text)


class FakeTurns:
    """Stands in for run_turn: records messages and blocks until released or cancelled."""

    def __init__(self):
        self.ran = []
        self.gate = threading.Event()
        self.started = threading.Event()
        self.hold_first = True
        self.approvals = []
        self.ask_approval = False

    def __call__(self, app, text, config, on_text, approve, cancel, on_speech=None, speech_limit=1200, notice=False):
        self.ran.append(text)
        self.started.set()
        on_text(f"reply:{text}")

        if self.ask_approval:
            request = {"calls": [{"name": "terminal", "args": {"command": "x"}}]}
            self.approvals.append(approve(request))

        if self.hold_first and len(self.ran) == 1:
            while not self.gate.is_set():
                if cancel.is_set():
                    raise TurnCancelled

                time.sleep(0.01)


class ChatLoopTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeTurns()
        patcher = patch.object(chat_loop, "run_turn", self.fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.out = []
        self.chat = ChatLoop(None, {}, JobManager(), write=self.out.append)
        self.chat.start()
        self.addCleanup(self.chat.close)

    @property
    def text(self):
        return "".join(self.out)

    def idle(self):
        return wait_for(lambda: not self.chat.busy)

    def test_messages_sent_mid_turn_are_queued_in_order(self):
        self.chat.submit("a")
        self.assertTrue(self.fake.started.wait(5))
        self.chat.submit("b")
        self.chat.submit("c")
        self.assertIn("Queued", self.text)
        self.assertEqual(self.fake.ran, ["a"])

        self.fake.gate.set()
        self.assertTrue(self.idle())
        self.assertEqual(self.fake.ran, ["a", "b", "c"])

    def test_stop_cancels_running_turn_and_drops_queue(self):
        self.chat.submit("a")
        self.assertTrue(self.fake.started.wait(5))
        self.chat.submit("b")
        self.chat.submit("stop")

        self.assertTrue(self.idle())
        self.assertEqual(self.fake.ran, ["a"])
        self.assertIn("[Stopped.]", self.text)

    def test_next_message_after_stop_runs_normally(self):
        self.chat.submit("a")
        self.assertTrue(self.fake.started.wait(5))
        self.chat.submit("cancel that")
        self.assertTrue(self.idle())

        self.chat.submit("b")
        self.assertTrue(self.idle())
        self.assertEqual(self.fake.ran, ["a", "b"])
        self.assertEqual(self.text.count("[Stopped.]"), 1)

    def test_request_mentioning_stop_is_queued_not_a_stop(self):
        self.chat.submit("a")
        self.assertTrue(self.fake.started.wait(5))
        self.chat.submit("stop the docker container")
        self.fake.gate.set()
        self.assertTrue(self.idle())
        self.assertEqual(self.fake.ran, ["a", "stop the docker container"])

    def test_stop_while_idle_is_harmless(self):
        self.chat.submit("stop")
        self.assertIn("Nothing is running", self.text)
        self.assertEqual(self.fake.ran, [])

    def test_quit(self):
        self.assertFalse(self.chat.submit("exit"))
        self.assertTrue(self.chat.submit(""))

    def test_job_notice_survives_stop_and_runs_afterwards(self):
        self.chat.submit("a")
        self.assertTrue(self.fake.started.wait(5))
        self.chat.submit("b")
        job = SimpleNamespace(id=7, command="pytest", report=lambda: "all passed")
        self.chat.job_finished(job)  # type: ignore[arg-type]
        self.chat.submit("stop")

        self.assertTrue(self.idle())
        self.assertEqual(len(self.fake.ran), 2)
        self.assertTrue(self.fake.ran[1].startswith(chat_loop.NOTICE_PREFIX))
        self.assertIn("all passed", self.fake.ran[1])

    def test_approval_answer_comes_from_the_same_input(self):
        self.fake.hold_first = False
        self.fake.ask_approval = True
        self.chat.submit("a")
        self.assertTrue(wait_for(lambda: self.chat._approval is not None))
        self.chat.submit("y")
        self.assertTrue(self.idle())
        self.assertEqual(self.fake.approvals, [True])

    def test_anything_but_yes_declines(self):
        self.fake.hold_first = False
        self.fake.ask_approval = True
        self.chat.submit("a")
        self.assertTrue(wait_for(lambda: self.chat._approval is not None))
        self.chat.submit("hmm maybe")
        self.assertTrue(self.idle())
        self.assertEqual(self.fake.approvals, [False])

    def test_stop_during_approval_cancels_turn(self):
        self.fake.hold_first = False
        self.fake.ask_approval = True
        self.chat.submit("a")
        self.assertTrue(wait_for(lambda: self.chat._approval is not None))
        self.chat.submit("stop")
        self.assertTrue(self.idle())
        self.assertEqual(self.fake.approvals, [])
        self.assertIn("[Stopped.]", self.text)


class RealGraphTurnTests(unittest.TestCase):
    """run_turn against the real graph with the model chains replaced."""

    def setUp(self):
        self.seen = []
        self.replies = []

        def code_chain(inputs):
            self.seen.append(list(inputs["messages"]))
            return self.replies.pop(0)

        router = RunnableLambda(lambda x: SimpleNamespace(destination="code"))

        for p in [
            patch.object(node, "router_chain", router),
            patch.object(node, "code_chain", RunnableLambda(code_chain)),
        ]:
            p.start()
            self.addCleanup(p.stop)

        self.app = ModelGraph.compile(checkpointer=InMemorySaver())
        self.config = {"configurable": {"thread_id": "t"}}

    def run_turn(self, text, cancel=None):
        out = []
        turns.run_turn(self.app, text, self.config, out.append, lambda r: True, cancel)
        return "".join(out)

    def test_turn_goes_through_the_existing_graph_and_keeps_history(self):
        self.replies = [AIMessage(content="one"), AIMessage(content="two")]
        self.assertEqual(self.run_turn("first"), "one")
        self.assertEqual(self.run_turn("second"), "two")
        self.assertEqual([m.content for m in self.seen[1]], ["first", "one", "second"])

    def test_cancel_raises_and_next_turn_still_works(self):
        self.replies = [AIMessage(content="never shown"), AIMessage(content="fine")]
        cancel = threading.Event()
        cancel.set()

        with self.assertRaises(TurnCancelled):
            self.run_turn("go", cancel)

        self.assertEqual(self.run_turn("again"), "fine")

    def test_tool_calls_left_open_by_a_cancelled_turn_are_answered(self):
        self.app.update_state(
            self.config,
            {
                "messages": [
                    HumanMessage(content="run tests"),
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "terminal", "args": {"command": "x"}, "id": "c1"}],
                    ),
                ]
            },
            as_node="code_tools",
        )
        self.replies = [AIMessage(content="ok")]

        self.assertEqual(self.run_turn("never mind, hello"), "ok")

        history = self.seen[0]
        results = [m for m in history if isinstance(m, ToolMessage)]
        self.assertEqual([m.tool_call_id for m in results], ["c1"])
        self.assertEqual(results[0].content, turns.CANCELLED_TOOL_OUTPUT)
        self.assertEqual(history[-1].content, "never mind, hello")


if __name__ == "__main__":
    unittest.main()
