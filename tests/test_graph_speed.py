"""F025: faster graph turns (timing log, context fitting, memory scheduling, routing)."""

import asyncio
import logging
import threading
import time
import unittest
from types import SimpleNamespace

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
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

    def test_a_history_limit_over_the_budget_is_warned_about(self):
        self.assertIsNone(node.history_budget_warning(5000, {}))
        warning = node.history_budget_warning(5000, {"LLM_NUM_CTX": "4096"})
        assert warning is not None
        self.assertIn("MAX_HISTORY_TOKENS", warning)
        self.assertIn("LLM_NUM_CTX", warning)

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


def load_chain_module(name: str):
    """chains/ is stubbed for the other tests, so load the real file directly."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "chains" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class SummaryModelTests(unittest.TestCase):
    """Loading a second model unloads qwen3:8b on a 4 GB GPU (reload ~8 s), so by default the
    summary uses the main model."""

    def setUp(self):
        self.module = load_chain_module("summerize_chain")

    def test_default_is_the_main_model(self):
        main = SimpleNamespace(model_copy=lambda update: ("copy of main", update))
        self.assertEqual(self.module.summary_model({}, main), ("copy of main", {"temperature": 0.3}))

    def test_summary_model_setting_still_picks_another_model(self):
        model = self.module.summary_model({"SUMMARY_MODEL": "qwen3:4b"}, object())
        self.assertEqual(model.model, "qwen3:4b")
        self.assertEqual(model.temperature, 0.3)
        self.assertFalse(model.reasoning)

    def test_blank_setting_means_the_main_model(self):
        main = SimpleNamespace(model_copy=lambda update: "main")
        self.assertEqual(self.module.summary_model({"SUMMARY_MODEL": "  "}, main), "main")


class MemorySchedulerTests(unittest.TestCase):
    """Extraction costs ~32 s on qwen3:8b and Ollama runs one generation at a time, so a save
    must not hold up the next turn: it waits for idle and is cancelled when a turn starts."""

    def scheduler(self, idle=0.05):
        from memories.scheduler import MemoryScheduler

        scheduler = MemoryScheduler(idle_seconds=idle)
        self.addCleanup(scheduler.wait, 5)
        return scheduler

    def test_a_save_waits_until_the_turn_is_over_and_the_chat_is_idle(self):
        scheduler = self.scheduler(idle=0.2)
        saved = threading.Event()
        manager = SimpleNamespace(remember=lambda messages, thread_id=None, cancel=None: saved.set())

        scheduler.turn_started()
        scheduler.submit(manager, ["m"], "t")
        self.assertFalse(saved.wait(0.3), "no save while a turn runs")

        scheduler.turn_finished()
        self.assertFalse(saved.wait(0.1), "no save right after the turn")
        self.assertTrue(saved.wait(2))

    def test_a_new_turn_cancels_the_running_save_and_it_is_retried_later(self):
        from memories.extractor import MemoryCancelled

        scheduler = self.scheduler()
        attempts, running, done = [], threading.Event(), threading.Event()

        def remember(messages, thread_id=None, cancel=None):
            attempts.append(1)
            running.set()
            if len(attempts) == 1:
                assert cancel is not None and cancel.wait(5), "the turn must cancel the save"
                raise MemoryCancelled()
            done.set()

        scheduler.submit(SimpleNamespace(remember=remember), ["m"], "t")
        self.assertTrue(running.wait(2))
        scheduler.turn_started()
        time.sleep(0.2)
        self.assertEqual(len(attempts), 1, "no retry while the turn runs")
        scheduler.turn_finished()

        self.assertTrue(done.wait(2))
        self.assertEqual(len(attempts), 2)

    def test_waiting_at_exit_does_not_wait_for_idle(self):
        scheduler = self.scheduler(idle=60)
        saved = []
        scheduler.turn_started()
        scheduler.turn_finished()
        scheduler.submit(SimpleNamespace(remember=lambda messages, thread_id=None, cancel=None: saved.append(1)), [], None)

        self.assertTrue(scheduler.wait(5))
        self.assertEqual(saved, [1])

    def test_a_failing_save_is_logged(self):
        scheduler = self.scheduler()

        def remember(messages, thread_id=None, cancel=None):
            raise RuntimeError("database down")

        with self.assertLogs("memories.scheduler", level="WARNING") as logged:
            scheduler.submit(SimpleNamespace(remember=remember), [], None)
            self.assertTrue(scheduler.wait(5))

        self.assertIn("Saving memory failed", logged.output[0])

    def test_run_turn_marks_the_turn_even_when_it_fails(self):
        import turns

        calls = []
        fake = SimpleNamespace(turn_started=lambda: calls.append("start"), turn_finished=lambda: calls.append("end"))
        app = SimpleNamespace(get_state=lambda config: (_ for _ in ()).throw(RuntimeError("boom")))

        with patch.object(turns, "MEMORY_SCHEDULER", fake), self.assertRaises(RuntimeError):
            turns.run_turn(app, "hi", {}, on_text=lambda t: None, approve=lambda p: True)

        self.assertEqual(calls, ["start", "end"])


class CancellableExtractionTests(unittest.TestCase):
    def test_cancel_stops_the_model_call(self):
        from memories.extractor import MemoryCancelled, invoke_cancellable

        stopped = threading.Event()

        async def slow(inputs):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                stopped.set()
                raise

        cancel = threading.Event()
        threading.Timer(0.1, cancel.set).start()
        started = time.monotonic()

        with self.assertRaises(MemoryCancelled):
            invoke_cancellable(RunnableLambda(lambda x: x, afunc=slow), {}, cancel)

        self.assertLess(time.monotonic() - started, 2)
        self.assertTrue(stopped.is_set(), "the request itself is cancelled (Ollama then stops)")

    def test_without_cancel_it_is_a_normal_call(self):
        from memories.extractor import invoke_cancellable

        def add_one(x: int) -> int:
            return x + 1

        self.assertEqual(invoke_cancellable(RunnableLambda(add_one), 1, None), 2)
        self.assertEqual(invoke_cancellable(RunnableLambda(add_one), 1, threading.Event()), 2)

    def test_calls_on_one_thread_share_an_open_event_loop(self):
        """ChatOllama keeps its async HTTP client (and pooled connections) from the first call;
        with a fresh, closed loop per call the second save failed with "Event loop is closed"."""
        from memories.extractor import invoke_cancellable

        loops = []

        async def record(inputs):
            loops.append(asyncio.get_running_loop())
            return inputs

        chain = RunnableLambda(lambda x: x, afunc=record)
        invoke_cancellable(chain, 1, threading.Event())
        invoke_cancellable(chain, 2, threading.Event())

        self.assertIs(loops[0], loops[1])
        self.assertFalse(loops[0].is_closed())

    def test_open_loops_are_closed_at_exit(self):
        """An event loop left open until interpreter shutdown logged "Cancelling an overlapped
        future failed" on Windows; `close_loops` runs at exit (atexit)."""
        from memories import extractor

        loops = []

        async def record(inputs):
            loops.append(asyncio.get_running_loop())
            return inputs

        chain = RunnableLambda(lambda x: x, afunc=record)
        extractor.invoke_cancellable(chain, 1, threading.Event())
        extractor.close_loops()
        self.assertTrue(loops[0].is_closed())

        extractor.invoke_cancellable(chain, 2, threading.Event())
        self.assertFalse(loops[1].is_closed(), "a later call gets a new loop")
        extractor.close_loops()

    def test_manager_hands_the_cancel_event_to_the_extractor(self):
        from memories.extractor import MemoryExtraction
        from memories.manager import MemoryManager

        seen = {}

        def extract(messages, known=(), cancel=None):
            seen["cancel"] = cancel
            return MemoryExtraction()

        store = SimpleNamespace(search=lambda *a, **k: SimpleNamespace(facts=[]), save_turn=lambda *a, **k: {})
        cancel = threading.Event()
        MemoryManager(store, SimpleNamespace(extract=extract)).remember(  # type: ignore[arg-type]
            [HumanMessage(content="I use Vim")], cancel=cancel
        )
        self.assertIs(seen["cancel"], cancel)


class StablePrefixTests(unittest.TestCase):
    """Ollama reuses its work only for an unchanged prompt start: a new memory text inside the
    system prompt made it re-read the whole history (8.0 s for 3,700 tokens). Ollama's qwen3
    template moves every system message to the top, so memory is a user message instead."""

    MEMORY = "Long-term memory:\n- The user's name is Dat."

    def history(self):
        return [
            HumanMessage(content="q1", id="h1"),
            AIMessage(content="a1", id="a1"),
            HumanMessage(content="q2", id="h2"),
            call("c1"),
            ToolMessage(content="body", tool_call_id="c1", id="t1"),
        ]

    def test_memory_goes_just_before_the_latest_user_message(self):
        messages = node.with_memory(self.history(), self.MEMORY)

        self.assertEqual([m.type for m in messages], ["human", "ai", "human", "human", "ai", "tool"])
        self.assertEqual(messages[2].content, self.MEMORY)

    def test_no_memory_adds_nothing(self):
        history = self.history()
        self.assertIs(node.with_memory(history, ""), history)

    def test_code_tools_and_direct_send_memory_as_a_message(self):
        for name in ["code_chain", "direct_chain"]:
            seen: dict = {}
            target = node.code_tools if name == "code_chain" else node.direct

            with patch.object(node, name, recorder(seen, AIMessage(content="ok"))):
                target({"messages": self.history(), "memory_context": self.MEMORY})  # type: ignore[arg-type]

            self.assertNotIn("memory", seen, name)
            sent = seen["messages"]
            self.assertFalse(any(isinstance(m, SystemMessage) for m in sent), "Ollama would hoist it")
            self.assertEqual([m.content for m in sent[2:4]], [self.MEMORY, "q2"], name)

    def test_prompt_start_does_not_depend_on_memory(self):
        for name, prompt_name in [("code_chain", "code_prompt"), ("direct_chain", "direct_prompt")]:
            prompt = getattr(load_chain_module(name), prompt_name)
            self.assertNotIn("memory", prompt.input_variables, name)

            first = prompt.format_messages(messages=node.with_memory(self.history(), "memory A"))
            second = prompt.format_messages(messages=node.with_memory(self.history(), "memory B"))
            self.assertEqual(first[:3], second[:3], "system prompt and earlier turns are identical")
            self.assertNotEqual(first[3], second[3])


class ParallelRouteTests(unittest.TestCase):
    """The router does not use recalled memory, so both start together."""

    def run_turn(self, destination, delay=0.4):
        from langgraph.checkpoint.memory import InMemorySaver

        from graph import ModelGraph

        def slow_route(inputs):
            time.sleep(delay)
            return SimpleNamespace(destination=destination)

        def slow_recall(query, context=""):
            time.sleep(delay)
            return "memory"

        seen: dict = {}
        manager = SimpleNamespace(recall=slow_recall, remember=lambda *a, **k: None)
        patches = [
            patch.object(node, "router_chain", RunnableLambda(slow_route)),
            patch.object(node, "direct_chain", recorder(seen, AIMessage(content="direct answer"))),
            patch.object(node, "code_chain", recorder({}, AIMessage(content="code answer"))),
            patch.dict(node.dependency, {"memory": manager}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        app = ModelGraph.compile(checkpointer=InMemorySaver())
        started = time.monotonic()
        final = app.invoke({"messages": [HumanMessage(content="tell me more")]}, {"configurable": {"thread_id": "p"}})  # type: ignore[arg-type]
        return time.monotonic() - started, final, seen

    def test_recall_and_routing_overlap(self):
        elapsed, final, _ = self.run_turn("direct")
        self.assertLess(elapsed, 0.75, "two 0.4 s steps in parallel, not one after the other")
        self.assertEqual(final["messages"][-1].content, "direct answer")

    def test_the_answer_still_gets_the_recalled_memory(self):
        _, _, seen = self.run_turn("direct", delay=0)
        self.assertIn("memory", [m.content for m in seen["messages"]])

    def test_code_route_reaches_the_coding_agent(self):
        _, final, _ = self.run_turn("code", delay=0)
        self.assertEqual(final["messages"][-1].content, "code answer")


class QuickRouteTests(unittest.TestCase):
    """Obvious requests skip the router's model call (3.0 s measured); anything unclear,
    including short follow-ups such as "yes, do it", still goes to the model."""

    CODE = [
        "fix the bug in main.py",
        "what does graph/node.py do?",
        "read README.md and summarize it",
        "run the tests",
        "git status",
        "show me the git diff",
        "pytest -k memory",
        "npm run build",
        "pip install requests",
        "delete the file old_notes.txt",
        "create a new file called notes.md",
        "search the repo for TODO",
        "sửa lỗi trong file main.py",
        "chạy test giúp mình",
        "tạo file mới tên notes.md",
    ]
    DIRECT = ["hi", "Hello!", "thanks", "Thank you.", "xin chào", "cảm ơn bạn", "good morning"]
    MODEL = [
        "yes, do it",
        "ok",
        "what is a python generator?",
        "explain recursion",
        "what is a .py file?",
        "Viết một bài thơ về mùa thu",
        "how do I write a file in python?",
    ]

    def test_obvious_coding_requests(self):
        for text in self.CODE:
            self.assertEqual(node.quick_route(text), "code_tools", text)

    def test_greetings_and_thanks(self):
        for text in self.DIRECT:
            self.assertEqual(node.quick_route(text), "direct", text)

    def test_everything_else_is_left_to_the_model(self):
        for text in self.MODEL:
            self.assertIsNone(node.quick_route(text), text)

    def test_router_skips_the_model_for_an_obvious_request(self):
        def fail(inputs):
            raise AssertionError("the model must not be asked")

        with patch.object(node, "router_chain", RunnableLambda(fail)):
            self.assertEqual(node.router({"messages": [HumanMessage(content="git status")]}), "code_tools")  # type: ignore[arg-type]

    def test_router_asks_the_model_when_unsure(self):
        reply = RunnableLambda(lambda x: SimpleNamespace(destination="code"))

        with patch.object(node, "router_chain", reply):
            self.assertEqual(node.router({"messages": [HumanMessage(content="yes, do it")]}), "code_tools")  # type: ignore[arg-type]

    def test_job_notices_are_left_to_the_model(self):
        notice = HumanMessage(content="[system notice] job 3 finished: git status", additional_kwargs={"notice": True})
        reply = RunnableLambda(lambda x: SimpleNamespace(destination="direct"))

        with patch.object(node, "router_chain", reply):
            self.assertEqual(node.router({"messages": [notice]}), "direct")  # type: ignore[arg-type]


class CleanupTests(unittest.TestCase):
    def test_no_empty_pass_through_node(self):
        from graph import ModelGraph

        self.assertNotIn("reverse_router", ModelGraph.nodes)

    def test_state_has_no_unused_fields(self):
        from data.state import State

        self.assertEqual(set(State.__annotations__), {"messages", "memory_context", "route"})

    def test_a_long_history_is_still_summarized_after_the_answer(self):
        from langgraph.checkpoint.memory import InMemorySaver

        from graph import ModelGraph

        long_reply = AIMessage(content=words(node.MAX_HISTORY_TOKENS + 10))
        patches = [
            patch.object(node, "router_chain", RunnableLambda(lambda x: SimpleNamespace(destination="direct"))),
            patch.object(node, "direct_chain", recorder({}, long_reply)),
            patch.object(node, "summerize_chain", RunnableLambda(lambda x: "SUMMARY")),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        app = ModelGraph.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "c"}}
        app.invoke({"messages": [HumanMessage(content="tell me")]}, config)  # type: ignore[arg-type]
        final = app.invoke({"messages": [HumanMessage(content="and more")]}, config)  # type: ignore[arg-type]

        self.assertTrue(final["messages"][0].content.endswith("SUMMARY"))


if __name__ == "__main__":
    unittest.main()
