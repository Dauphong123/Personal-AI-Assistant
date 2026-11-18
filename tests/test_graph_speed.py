"""F025: faster graph turns (timing log, context fitting, memory scheduling, routing)."""

import logging
import unittest
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

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


if __name__ == "__main__":
    unittest.main()
