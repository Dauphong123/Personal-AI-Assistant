"""Opt-in evaluation of the main model on routing and tool choice (real Ollama).

    $env:RUN_MODEL_EVAL = "1"; python -m unittest tests.test_model_eval -v
    $env:RUN_MODEL_EVAL = "1"; $env:EVAL_MODEL = "qwen3:4b"; python -m unittest tests.test_model_eval -v

Compares models before switching MODEL (F025): the router's model call (the rule shortcut is
bypassed on purpose) and the first tool the coding agent picks. Prints accuracy and speed.
"""

import importlib.util
import os
import sys
import time
import types
import unittest
from pathlib import Path

from langchain_core.messages import HumanMessage

from tests import stubs

RUN = os.environ.get("RUN_MODEL_EVAL") == "1"
MODEL = os.environ.get("EVAL_MODEL", "qwen3:8b")
MIN_ACCURACY = 0.8
ROOT = Path(__file__).resolve().parents[1]

ROUTING = [
    ("what is a python generator?", "direct"),
    ("explain the difference between TCP and UDP", "direct"),
    ("Viết một câu chào buổi sáng", "direct"),
    ("what's the capital of Australia?", "direct"),
    ("summarize the idea of dependency injection in two sentences", "direct"),
    ("translate 'good night' to French", "direct"),
    ("which functions in this project call the router?", "code"),
    ("add a docstring to the summarize node", "code"),
    ("are there uncommitted changes?", "code"),
    ("how many tests does this repository have?", "code"),
    ("mình muốn xem lịch sử commit gần đây", "code"),
    ("check whether ripgrep is installed on this machine", "code"),
]

TOOLS = [
    ("what files are in the memories folder?", {"list_files"}),
    ("where is MAX_HISTORY_TOKENS defined?", {"search_code"}),
    ("show me the last 3 commits", {"git_log"}),
    ("what changed in the working tree?", {"git_status", "git_diff"}),
    ("read llm_options.py and tell me what reasoning does", {"read_file"}),
    ("search my knowledge documents for chroma", {"search_document"}),
    ("run python --version", {"terminal"}),
    ("mở file README.md và cho mình biết dự án làm gì", {"read_file"}),
]


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"eval_{name}", ROOT / "chains" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@unittest.skipUnless(RUN, "set RUN_MODEL_EVAL=1 (needs Ollama with EVAL_MODEL pulled)")
class ModelEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from langchain_ollama import ChatOllama

        from llm_options import ollama_options

        stubs.install()
        model = ChatOllama(model=MODEL, **ollama_options())
        sys.modules["dependency"].dependency["model"] = model  # type: ignore[attr-defined]
        cls.router_chain = load("router_chain").router_chain
        cls.code_chain = load("code_chain").code_chain
        model.invoke("hi")  # load the model before timing

    def score(self, cases, ask) -> float:
        right = 0
        started = time.monotonic()

        for question, expected in cases:
            got = ask(question)
            ok = got in expected if isinstance(expected, set) else got == expected
            right += ok
            print(f"  {'ok ' if ok else 'BAD'} {question!r} -> {got}", flush=True)

        accuracy = right / len(cases)
        print(f"{MODEL}: {right}/{len(cases)} in {time.monotonic() - started:.0f}s", flush=True)
        return accuracy

    def test_routing(self):
        def ask(question):
            return self.router_chain.invoke({"question": question, "history": ""}).destination

        self.assertGreaterEqual(self.score(ROUTING, ask), MIN_ACCURACY)

    def test_first_tool_choice(self):
        def ask(question):
            reply = self.code_chain.invoke({"messages": [HumanMessage(content=question)]})
            calls = reply.tool_calls
            return calls[0]["name"] if calls else "(no tool)"

        self.assertGreaterEqual(self.score(TOOLS, ask), MIN_ACCURACY)


if __name__ == "__main__":
    unittest.main()
