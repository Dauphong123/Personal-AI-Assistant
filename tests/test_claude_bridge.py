"""F036: running Claude Code headless (stream-json), with no real `claude` process: a fake one replays
the fixtures in tests/fixtures/claude_stream (shapes assumed from the Agent SDK protocol)."""

import io
import json
import queue
import threading
import time
import unittest
from pathlib import Path

from claude_bridge import (
    APPROVAL_TOOLS,
    ClaudeCodeChat,
    ClaudeCodeError,
    Done,
    Permission,
    Text,
    ToolResult,
    ToolStart,
    claude_command,
    parse_line,
    run_claude,
)
from turns import TurnCancelled

FIXTURES = Path(__file__).parent / "fixtures" / "claude_stream"


def fixture(name: str) -> list[str]:
    return (FIXTURES / name).read_text(encoding="utf-8").splitlines(keepends=True)


class FakeStdin(io.StringIO):
    def __init__(self):
        super().__init__()
        self.lines: list[dict] = []

    def write(self, s: str) -> int:
        for line in s.splitlines():
            if line.strip():
                self.lines.append(json.loads(line))
        return len(s)

    def flush(self) -> None:
        pass


class FakeProcess:
    """stdout lines come from a queue; None ends the stream."""

    def __init__(self, lines: list[str] | None = None, returncode: int = 0, stderr: str = ""):
        self.out: queue.Queue[str | None] = queue.Queue()
        for line in lines or []:
            self.out.put(line)
        if lines is not None:
            self.out.put(None)
        self.stdin = FakeStdin()
        self.stderr = io.StringIO(stderr)
        self.returncode = returncode
        self.killed = False
        self.pid = 4242

    @property
    def stdout(self):
        while True:
            line = self.out.get()
            if line is None:
                return
            yield line

    def wait(self, timeout=None) -> int:
        return self.returncode


class Popen:
    def __init__(self, process: FakeProcess):
        self.process = process
        self.kwargs: dict = {}

    def __call__(self, args, **kwargs):
        self.kwargs = kwargs
        return self.process


def killer(process) -> None:
    process.killed = True
    process.out.put(None)


def run(process: FakeProcess, approve=lambda request: True, cancel: threading.Event | None = None):
    return list(
        run_claude(
            ["claude", "-p"],
            "hello",
            "D:/project",
            approve,
            cancel or threading.Event(),
            popen=Popen(process),
            kill=killer,
        )
    )


def make_chat(
    runner, alias: str = "haiku", effort: str = "low", cwd: str = ".", resume: bool = False
) -> ClaudeCodeChat:
    return ClaudeCodeChat(
        alias=alias,
        effort=effort,
        cwd=cwd,
        session_id="S",
        resume=resume,
        approve=lambda request: True,
        cancel=threading.Event(),
        runner=runner,
    )


class CommandTests(unittest.TestCase):
    def test_command_flags(self):
        command = claude_command("sonnet", "medium", "U", False)

        def value(flag):
            return command[command.index(flag) + 1]

        self.assertEqual(command[:2], ["claude", "-p"])
        self.assertEqual(value("--model"), "sonnet")
        self.assertEqual(value("--effort"), "medium")
        self.assertEqual(value("--setting-sources"), "project,local")
        self.assertEqual(value("--permission-prompts"), "host")
        # Without it Claude Code denies edits itself ("permission_denied", source config) and never asks us.
        self.assertEqual(value("--permission-prompt-tool"), "stdio")
        self.assertEqual(value("--input-format"), "stream-json")
        self.assertEqual(value("--output-format"), "stream-json")
        self.assertIn("--strict-mcp-config", command)
        self.assertIn("--include-partial-messages", command)
        self.assertEqual(value("--session-id"), "U")
        self.assertNotIn("--resume", command)

        resumed = claude_command("sonnet", "medium", "U", True, exe="C:/bin/claude.cmd")
        self.assertEqual(resumed[0], "C:/bin/claude.cmd")
        self.assertEqual(resumed[resumed.index("--resume") + 1], "U")
        self.assertNotIn("--session-id", resumed)

    def test_fallback_model(self):
        # F038 step 3: Claude Code itself switches when the main model is busy or unavailable.
        command = claude_command("sonnet", "medium", "U", False, fallback="haiku,opus")
        self.assertEqual(command[command.index("--fallback-model") + 1], "haiku,opus")

        self.assertNotIn("--fallback-model", claude_command("sonnet", "medium", "U", False))
        # a fallback equal to the main model is left out
        self.assertNotIn("--fallback-model", claude_command("haiku", "medium", "U", False, fallback="haiku"))
        same_dropped = claude_command("haiku", "medium", "U", False, fallback="haiku,sonnet")
        self.assertEqual(same_dropped[same_dropped.index("--fallback-model") + 1], "sonnet")

    def test_fallback_setting(self):
        from settings import check_settings, claude_fallback_model

        self.assertEqual(claude_fallback_model({}), "")
        self.assertEqual(claude_fallback_model({"CLAUDE_FALLBACK_MODEL": " haiku, sonnet "}), "haiku,sonnet")
        for raw in ("haiku;rm -rf", "--model", "a,,b"):
            with self.subTest(raw=raw):
                problems = check_settings({"MODEL": "qwen3:8b", "CLAUDE_FALLBACK_MODEL": raw})
                self.assertTrue(any("CLAUDE_FALLBACK_MODEL" in p for p in problems), problems)

    def test_approval_tools(self):
        self.assertEqual(APPROVAL_TOOLS, frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit", "Bash"}))


class ParseTests(unittest.TestCase):
    def parse(self, name: str) -> list:
        return [event for line in fixture(name) for event in parse_line(line)]

    def test_text_comes_from_deltas_once(self):
        events = self.parse("text.jsonl")
        text = "".join(e.text for e in events if isinstance(e, Text))
        self.assertEqual(text, "Đã sửa xong.")
        self.assertEqual(events[-1], Done("Đã sửa xong.", False))

    def test_tool_calls_and_results(self):
        events = self.parse("tool_read.jsonl")
        self.assertIn(ToolStart("toolu_1", "Read", {"file_path": "notes.txt"}), events)
        self.assertIn(ToolResult("toolu_1", "The secret word is pelican.", False), events)

    def test_permission_requests(self):
        asks = [e for e in self.parse("permission.jsonl") if isinstance(e, Permission)]
        self.assertEqual(
            asks,
            [
                Permission("req_1", "Glob", {"pattern": "*.txt"}),
                Permission("req_2", "Write", {"file_path": "made.txt", "content": "done"}),
            ],
        )

    def test_error_result(self):
        self.assertEqual(self.parse("error.jsonl")[-1], Done("Claude AI usage limit reached. Try again later.", True))

    def test_junk_is_ignored(self):
        self.assertEqual(parse_line("not json"), [])
        self.assertEqual(parse_line("[1, 2]"), [])
        self.assertEqual(parse_line('{"type": "unknown"}'), [])

    def test_system_lines_with_a_text_message_are_skipped(self):
        line = json.dumps({"type": "system", "subtype": "permission_denied", "tool_name": "Write", "message": "not granted"})
        self.assertEqual(parse_line(line), [])

    def test_real_claude_code_run(self):
        # Trimmed capture of claude 2.1.330 (haiku): asked to write scratch.txt, the host denied it.
        events = self.parse("real_write_denied.jsonl")
        asks = [e for e in events if isinstance(e, Permission)]
        self.assertEqual([(a.tool, a.args) for a in asks], [("Write", {"file_path": r"D:\app\Langchain\scratch.txt", "content": "hello"})])
        starts = [e for e in events if isinstance(e, ToolStart)]
        results = [e for e in events if isinstance(e, ToolResult)]
        self.assertEqual([s.name for s in starts], ["Write"])
        self.assertEqual([(r.id, r.is_error) for r in results], [(starts[0].id, True)])
        text = "".join(e.text for e in events if isinstance(e, Text))
        self.assertTrue(text.startswith("I didn't create `scratch.txt`"))
        self.assertIsInstance(events[-1], Done)
        self.assertFalse(events[-1].is_error)
        self.assertEqual(events[-1].text, text)

    def test_tool_result_with_block_content(self):
        block = {"type": "tool_result", "tool_use_id": "t", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
        line = json.dumps({"type": "user", "message": {"content": [block]}})
        self.assertEqual(parse_line(line), [ToolResult("t", "ab", False)])


class RunTests(unittest.TestCase):
    def test_prompt_is_sent_as_a_user_line(self):
        process = FakeProcess(fixture("text.jsonl"))
        run(process)
        self.assertEqual(process.stdin.lines[0], {"type": "user", "message": {"role": "user", "content": "hello"}})

    def test_read_tools_are_allowed_without_asking(self):
        asked: list[dict] = []

        def approve(request: dict) -> bool:
            asked.append(request)
            return True

        process = FakeProcess(fixture("permission.jsonl"))
        events = run(process, approve=approve)

        answers = {a["response"]["request_id"]: a["response"] for a in process.stdin.lines if a["type"] == "control_response"}
        self.assertEqual(answers["req_1"]["response"], {"behavior": "allow", "updatedInput": {"pattern": "*.txt"}})
        self.assertEqual(answers["req_2"]["response"]["behavior"], "allow")
        self.assertEqual(asked, [{"calls": [{"name": "Write", "args": {"file_path": "made.txt", "content": "done"}}]}])
        self.assertFalse(any(isinstance(e, Permission) for e in events))

    def test_edit_asks_and_deny_is_sent(self):
        process = FakeProcess(fixture("permission.jsonl"))
        run(process, approve=lambda request: False)

        answers = {a["response"]["request_id"]: a["response"] for a in process.stdin.lines if a["type"] == "control_response"}
        self.assertEqual(answers["req_2"]["subtype"], "success")
        self.assertEqual(answers["req_2"]["response"]["behavior"], "deny")
        self.assertIn("declined", answers["req_2"]["response"]["message"])

    def test_events_stream_in_order(self):
        kinds = [type(e).__name__ for e in run(FakeProcess(fixture("tool_read.jsonl")))]
        self.assertEqual(kinds, ["Text", "ToolStart", "ToolResult", "Text", "Done"])

    def test_cancel_kills_and_raises(self):
        process = FakeProcess()  # never ends by itself
        cancel = threading.Event()
        threading.Timer(0.2, cancel.set).start()
        started = time.monotonic()

        with self.assertRaises(TurnCancelled):
            run(process, cancel=cancel)

        self.assertTrue(process.killed)
        self.assertLess(time.monotonic() - started, 2)

    def test_cancel_while_approval_waits(self):
        process = FakeProcess()
        process.out.put(fixture("permission.jsonl")[5])  # the Write request
        cancel = threading.Event()

        def approve(request: dict) -> bool:
            cancel.set()  # the user pressed Stop while asked
            return False

        with self.assertRaises(TurnCancelled):
            run(process, approve=approve, cancel=cancel)

        self.assertTrue(process.killed)

    def test_stop_raised_by_the_approval_kills_claude(self):
        # The real approve callbacks (ChatLoop, ChatSession) raise TurnCancelled on Stop.
        process = FakeProcess()
        process.out.put(fixture("permission.jsonl")[5])

        def approve(request: dict) -> bool:
            raise TurnCancelled

        with self.assertRaises(TurnCancelled):
            run(process, approve=approve)

        self.assertTrue(process.killed)

    def test_any_failure_mid_turn_kills_claude(self):
        process = FakeProcess()
        process.out.put(fixture("permission.jsonl")[5])

        def approve(request: dict) -> bool:
            raise RuntimeError("approval box broke")

        with self.assertRaises(RuntimeError):
            run(process, approve=approve)

        self.assertTrue(process.killed)

    def test_missing_claude_is_a_clear_error(self):
        def popen(args, **kwargs):
            raise FileNotFoundError("claude")

        with self.assertRaisesRegex(ClaudeCodeError, "not installed"):
            list(run_claude(["claude"], "hi", ".", lambda r: True, threading.Event(), popen=popen))

    def test_error_result_is_raised(self):
        with self.assertRaisesRegex(ClaudeCodeError, "usage limit reached"):
            run(FakeProcess(fixture("error.jsonl")))

    def test_exit_without_result_is_one_sentence_and_logs_stderr(self):
        stderr = "Error: unknown option '--effort'\n    at Command.parse (cli.js:1:2)\n    at main (cli.js:3:4)\n"
        process = FakeProcess([], returncode=1, stderr=stderr)

        with self.assertLogs("claude_bridge", "WARNING") as logs, self.assertRaises(ClaudeCodeError) as caught:
            run(process)

        message = str(caught.exception)
        self.assertNotIn("\n", message)
        self.assertNotIn("cli.js", message)
        self.assertIn("exit code 1", message)
        self.assertIn("unknown option", "\n".join(logs.output))

    def test_error_result_shows_only_its_first_line(self):
        line = json.dumps({"type": "result", "is_error": True, "result": "Not logged in.\nRun claude and /login.\nDetails: x"})
        with self.assertRaises(ClaudeCodeError) as caught:
            run(FakeProcess([line + "\n"]))
        self.assertEqual(str(caught.exception), "Not logged in.")

    def test_vietnamese_round_trip(self):
        process = FakeProcess(fixture("text.jsonl"))
        popen = Popen(process)
        events = list(run_claude(["claude"], "Sửa lỗi trong file này", ".", lambda r: True, threading.Event(), popen=popen))
        self.assertEqual(process.stdin.lines[0]["message"]["content"], "Sửa lỗi trong file này")
        self.assertEqual(popen.kwargs.get("encoding"), "utf-8")
        self.assertIn(Text("Đã sửa"), events)


class ChatModelTests(unittest.TestCase):
    def test_chat_model_streams_text_and_tool_chunks(self):
        seen: dict = {}

        def runner(command, prompt, cwd, approve, cancel):
            seen.update(command=command, prompt=prompt, cwd=cwd)
            yield Text("Hi")
            yield ToolStart("t1", "Read", {"file_path": "a.py"})
            yield ToolResult("t1", "print(1)", False)
            yield Text(" there")
            yield Done("Hi there", False)

        chat = make_chat(runner, alias="sonnet", effort="high", cwd="D:/project", resume=True)
        # LangChain ends a stream with one empty closing chunk of its own; skip it.
        chunks = [c for c in chat.stream("What is in a.py?") if c.content or c.additional_kwargs]

        self.assertEqual([c.content for c in chunks], ["Hi", "", "", " there"])
        tools = [c.additional_kwargs["claude_tools"][0] for c in chunks if "claude_tools" in c.additional_kwargs]
        self.assertEqual([t["status"] for t in tools], ["running", "done"])
        self.assertEqual(tools[0]["name"], "Read")
        self.assertEqual(tools[1]["text"], "print(1)")
        self.assertEqual(seen["prompt"], "What is in a.py?")
        self.assertEqual(seen["cwd"], "D:/project")
        self.assertEqual(seen["command"][seen["command"].index("--resume") + 1], "S")
        self.assertEqual(seen["command"][seen["command"].index("--effort") + 1], "high")

    def test_tool_chunks_add_up_without_merging(self):
        # LangChain adds chunks together; two calls with the same argument of different values
        # must not clash (merge_dicts raises on differing booleans).
        def runner(command, prompt, cwd, approve, cancel):
            yield ToolStart("t1", "Edit", {"file_path": "a.py", "replace_all": False})
            yield ToolStart("t2", "Edit", {"file_path": "b.py", "replace_all": True})
            yield ToolResult("t1", "ok", False)
            yield Done("", False)

        chunks = list(make_chat(runner).stream("x"))
        total = chunks[0]
        for chunk in chunks[1:]:
            total = total + chunk
        tools = total.additional_kwargs["claude_tools"]
        self.assertEqual([t["id"] for t in tools], ["t1", "t2", "t1"])
        self.assertEqual(tools[1]["args"]["replace_all"], True)

    def test_failed_tool_is_reported_as_error(self):
        def runner(command, prompt, cwd, approve, cancel):
            yield ToolStart("t1", "Write", {})
            yield ToolResult("t1", "The user declined this action.", True)
            yield Done("", False)

        tools = [c.additional_kwargs["claude_tools"][0] for c in make_chat(runner).stream("x") if c.additional_kwargs]
        self.assertEqual(tools[-1]["status"], "error")
        self.assertEqual(tools[-1]["name"], "Write")

    def test_empty_answer_still_returns_a_message(self):
        def runner(command, prompt, cwd, approve, cancel):
            yield Done("", False)

        self.assertEqual(make_chat(runner).invoke("x").content, "")
        # LangGraph streams the model; a stream with no chunk at all is an error in LangChain.
        chunks = list(make_chat(runner).stream("x"))
        self.assertTrue(chunks)
        self.assertEqual("".join(str(c.content) for c in chunks), "")

    def test_invoke_joins_the_text(self):
        def runner(command, prompt, cwd, approve, cancel):
            yield Text("a")
            yield Text("b")
            yield Done("ab", False)

        self.assertEqual(make_chat(runner).invoke("x").content, "ab")


if __name__ == "__main__":
    unittest.main()
