"""Opt-in evaluation of the main model on routing and tool choice (real Ollama).

    $env:RUN_MODEL_EVAL = "1"; python -m unittest tests.test_model_eval -v
    $env:RUN_MODEL_EVAL = "1"; $env:EVAL_MODEL = "qwen3:4b"; python -m unittest tests.test_model_eval -v

Compares models before switching MODEL (F025): the router's model call (the rule shortcut is
bypassed on purpose) and the first tool the coding agent picks. Prints accuracy and speed.
F027: the first call's arguments as the tools receive them (qwen3:8b 6/6 on 2026-11-18; without
F027's repairs 2/6: JSON content arrived as a dict, "/voice" resolved to the drive root).
Set PYTHONIOENCODING=utf-8 when the output is piped (the Vietnamese case is printed).
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

# F027: the first call's arguments after `tool_args.prepare` (JSON content arrives as a dict from
# langchain_ollama, directories have four names). Expected values are a subset of the arguments.
ARGUMENTS = [
    ('create the file notes/settings.json containing {"debug": true}',
     "write_file", {"path": "notes/settings.json", "content": {"debug": True}}),
    ("show the git status of the memories folder", "git_status", {"dir": "memories"}),
    ("list the files in the voice directory", "list_files", {"path": "voice"}),
    ("search for MAX_HISTORY_TOKENS in the graph folder", "search_code", {"directory": "graph"}),
    ("read lines 1 to 20 of main.py", "read_file", {"path": "main.py", "start_line": 1, "end_line": 20}),
    ("show the last 3 commits of the memories folder", "git_log", {"dir": "memories", "limit": 3}),
]  # fmt: skip


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

    def test_tool_arguments(self):
        """Prints how many calls only work because tool_args repaired them."""
        import json

        import tool_args
        import tools

        repaired = 0

        def matches(name: str, args: dict, wanted: tuple[str, dict]) -> bool:
            tool_name, wanted_args = wanted
            if name != tool_name:
                return False
            for key, value in wanted_args.items():
                got = args.get(key)
                if isinstance(value, dict):  # JSON file content: compare parsed
                    try:
                        got = json.loads(got) if isinstance(got, str) else None
                    except ValueError:
                        return False
                elif isinstance(value, int) and isinstance(got, str) and got.isdigit():
                    got = int(got)
                elif key in {"path", "dir", "directory", "repo_path"} and isinstance(got, str):
                    try:  # what the tool will open, so "/voice" and "voice" both count
                        same = tools.resolve_path(got) == tools.resolve_path(str(value))
                    except tools.PathOutsideProjectError:
                        same = False
                    if not same:
                        return False
                    continue
                if got != value:
                    return False
            return True

        def ask(question):
            nonlocal repaired
            reply = self.code_chain.invoke({"messages": [HumanMessage(content=question)]})
            if not reply.tool_calls:
                return "(no tool)"
            call = reply.tool_calls[0]
            tool = tools.TOOL_BY_NAME.get(call["name"])
            if tool is None:
                return f"(unknown {call['name']})"
            args, error = tool_args.prepare(tool, call["args"])
            repaired += error is None and args != call["args"]
            print(f"      raw {call['name']}{call['args']} -> {error or args}", flush=True)
            return "ok" if error is None and matches(call["name"], args, expected[question]) else "bad"

        expected = {question: (name, args) for question, name, args in ARGUMENTS}
        cases = [(question, "ok") for question, _, _ in ARGUMENTS]
        accuracy = self.score(cases, ask)
        print(f"  repaired by tool_args: {repaired}/{len(cases)}", flush=True)
        self.assertGreaterEqual(accuracy, MIN_ACCURACY)


if __name__ == "__main__":
    unittest.main()
