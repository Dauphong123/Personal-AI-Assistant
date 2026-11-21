"""F036: the model chosen for a turn reaches the graph (Ollama models other than MODEL)."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from tests import stubs

stubs.install()

import model_chains  # noqa: E402
import turns  # noqa: E402
from graph import ModelGraph, node  # noqa: E402
from model_choice import ModelChoice  # noqa: E402


def answering(text: str, calls: list[str]):
    def reply(_):
        calls.append(text)
        return AIMessage(content=text)

    return RunnableLambda(reply)


class GraphModelTests(unittest.TestCase):
    def setUp(self):
        self.calls: list[str] = []
        patches = [
            patch.dict("os.environ", {"MODEL": "qwen3:8b"}),
            patch.object(node, "router_chain", RunnableLambda(lambda x: SimpleNamespace(destination="direct"))),
            patch.object(node, "direct_chain", answering("main", self.calls)),
            patch.object(
                node,
                "chains_for",
                lambda name: (answering(name, self.calls), answering(name + " code", self.calls)),
            ),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.app = ModelGraph.compile(checkpointer=InMemorySaver())

    def ask(self, model: ModelChoice | None) -> str:
        configurable: dict = {"thread_id": "t"}
        if model is not None:
            configurable["model"] = model
        result = self.app.invoke({"messages": [HumanMessage(content="hi there friend")]}, {"configurable": configurable})  # type: ignore[arg-type]
        return result["messages"][-1].content

    def test_chosen_ollama_model_answers(self):
        self.assertEqual(self.ask(ModelChoice("qwen3:4b", None)), "qwen3:4b")
        self.assertEqual(self.calls, ["qwen3:4b"])

    def test_no_choice_uses_the_main_chains(self):
        self.assertEqual(self.ask(None), "main")

    def test_choosing_model_itself_uses_the_main_chains(self):
        self.assertEqual(self.ask(ModelChoice("qwen3:8b", None)), "main")

    def test_turn_model_reads_the_config(self):
        choice = ModelChoice("qwen3:4b", None)
        self.assertEqual(node.turn_model({"configurable": {"model": choice}}), choice)
        self.assertIsNone(node.turn_model({"configurable": {}}))
        self.assertIsNone(node.turn_model({}))


class ChainsForTests(unittest.TestCase):
    def test_chains_are_built_once_per_model(self):
        built: list[str] = []

        def build(name):
            built.append(name)
            return (object(), object())

        with patch.object(model_chains, "_build", build), patch.dict(model_chains._CACHE, clear=True):
            first = model_chains.chains_for("qwen3:4b")
            again = model_chains.chains_for("qwen3:4b")
            model_chains.chains_for("llama3")

        self.assertIs(first, again)
        self.assertEqual(built, ["qwen3:4b", "llama3"])


class RunTurnConfigTests(unittest.TestCase):
    def test_run_turn_puts_model_approve_and_cancel_in_the_config(self):
        seen: list[dict] = []

        class App:
            def stream(self, graph_input, config, stream_mode):
                seen.append(config)
                yield from ()

            def get_state(self, config):
                return SimpleNamespace(tasks=[], values={"messages": []})

        def approve(request: dict) -> bool:
            return True

        cancel = threading.Event()
        choice = ModelChoice("claude-code:opus", "high")
        config = {"configurable": {"thread_id": "1"}}

        turns.run_turn(App(), "hello", config, lambda text: None, approve, cancel=cancel, model=choice)

        configurable = seen[0]["configurable"]
        self.assertEqual(configurable["thread_id"], "1")
        self.assertIs(configurable["model"], choice)
        self.assertIs(configurable["approve"], approve)
        self.assertIs(configurable["cancel"], cancel)
        self.assertNotIn("model", config["configurable"], "the caller's config is not changed")


if __name__ == "__main__":
    unittest.main()
