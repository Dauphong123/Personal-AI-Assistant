"""F025: faster graph turns (timing log, context fitting, memory scheduling, routing)."""

import logging
import unittest
from types import SimpleNamespace

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from tests import stubs

stubs.install()

from graph import node


class TimingLogTests(unittest.TestCase):
    def test_timed_node_logs_its_name_and_duration(self):
        @node.timed
        def sample(state):
            return {"ok": state}

        with self.assertLogs("graph.timing", level="INFO") as logged:
            self.assertEqual(sample(1), {"ok": 1})

        self.assertRegex(logged.output[0], r"sample \d+\.\d+s")

    def test_timed_keeps_the_signature_langgraph_inspects(self):
        import inspect

        @node.timed
        def with_config(state, config):
            return {}

        self.assertEqual(list(inspect.signature(with_config).parameters), ["state", "config"])

    def test_ollama_stats_describe_prompt_and_output(self):
        message = AIMessage(
            content="x",
            response_metadata={
                "prompt_eval_count": 3701,
                "prompt_eval_duration": 2_900_000_000,
                "eval_count": 55,
                "eval_duration": 10_000_000_000,
            },
        )
        stats = node.ollama_stats(message)
        self.assertIn("prompt 3701 tok in 2.9s", stats)
        self.assertIn("output 55 tok at 5.5 tok/s", stats)

    def test_ollama_stats_empty_without_metadata(self):
        self.assertEqual(node.ollama_stats(AIMessage(content="x")), "")
        self.assertEqual(node.ollama_stats(SimpleNamespace()), "")

    def test_timing_is_off_unless_enabled(self):
        self.assertEqual(node.timing_level({}), logging.WARNING)
        self.assertEqual(node.timing_level({"LOG_TIMING": "true"}), logging.INFO)
        with self.assertRaises(ValueError):
            node.timing_level({"LOG_TIMING": "maybe"})


def call(id):
    return AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "x"}, "id": id}])


def words(n, tag="w"):
    return " ".join(f"{tag}{i}" for i in range(n))


def recorder(seen: dict, reply):
    def run(inputs: dict):
        seen.update(inputs)
        return reply

    return RunnableLambda(run)


class FitToContextTests(unittest.TestCase):
    """The stub tokenizer counts words, so budgets here are in words."""

    def turn(self):
        return [
            HumanMessage(content="read both", id="h"),
            call("c1"),
            ToolMessage(content=words(400, "old"), tool_call_id="c1", id="t1"),
            call("c2"),
            ToolMessage(content=words(400, "new"), tool_call_id="c2", id="t2"),
        ]

    def test_messages_that_fit_are_returned_unchanged(self):
        messages = self.turn()
        self.assertIs(node.fit_to_context(messages, budget=10_000), messages)

    def test_oldest_tool_result_is_trimmed_first(self):
        messages = self.turn()
        fitted = node.fit_to_context(messages, budget=500)

        self.assertLessEqual(node.count_message_tokens(fitted), 500)
        self.assertIn("trimmed", fitted[2].content)
        self.assertEqual(fitted[4].content, messages[4].content, "the newest result stays whole")
        self.assertEqual([m.tool_call_id for m in fitted if isinstance(m, ToolMessage)], ["c1", "c2"])
        self.assertEqual(len(messages[2].content.split()), 400, "the state is not changed")

    def test_newest_result_is_cut_when_trimming_the_older_ones_is_not_enough(self):
        fitted = node.fit_to_context(self.turn(), budget=200)

        self.assertLessEqual(node.count_message_tokens(fitted), 200)
        self.assertTrue(fitted[4].content.startswith("new0 new1"))
        self.assertIn("trimmed", fitted[4].content)

    def test_budget_comes_from_the_context_window(self):
        self.assertEqual(
            node.message_budget({}),
            8192 - node.TOOL_PROMPT_TOKENS - node.ANSWER_RESERVE_TOKENS,
        )
        self.assertEqual(
            node.message_budget({"LLM_NUM_CTX": "16384"}),
            16384 - node.TOOL_PROMPT_TOKENS - node.ANSWER_RESERVE_TOKENS,
        )

    def test_history_limit_leaves_room_for_a_tool_result(self):
        self.assertLess(node.MAX_HISTORY_TOKENS, node.message_budget({}))

    def test_code_tools_sends_the_fitted_messages(self):
        seen: dict = {}
        chain = recorder(seen, AIMessage(content="ok"))

        with patch.object(node, "code_chain", chain), patch.object(node, "MESSAGE_BUDGET", 500):
            node.code_tools({"messages": self.turn()})  # type: ignore[arg-type]

        self.assertLessEqual(node.count_message_tokens(seen["messages"]), 500)

    def test_direct_sends_the_fitted_messages(self):
        seen: dict = {}
        chain = recorder(seen, AIMessage(content="ok"))
        messages = self.turn() + [AIMessage(content="done"), HumanMessage(content="thanks")]

        with patch.object(node, "direct_chain", chain), patch.object(node, "MESSAGE_BUDGET", 500):
            node.direct({"messages": messages})  # type: ignore[arg-type]

        self.assertLessEqual(node.count_message_tokens(seen["messages"]), 500)


if __name__ == "__main__":
    unittest.main()
