# F036 Claude Code Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user pick the chat model per session (installed Ollama models or Claude Code with an effort level), with Claude Code turns run headless through the user's own `claude` login.

**Architecture:** `model_choice.py` owns model ids, the live list and `model_settings.json`. `claude_bridge.py` runs `claude -p` in stream-json mode, answers its permission requests through our `approve` callback, and exposes it as a LangChain chat model so LangGraph's `messages` stream carries its text like any other model. The graph gets one node, `claude_code`, chosen by `route` from the run config; CLI, backend and frontend pass the choice into `run_turn`.

**Tech Stack:** Python 3.12 stdlib (`subprocess`, `json`, `uuid`, `urllib`), LangChain `BaseChatModel`, LangGraph, TypeScript frontend (`tsc`, `node --test`).

**Spec:** `docs/superpowers/specs/2026-11-20-f036-claude-code-bridge-design.md`

## Global Constraints

- No new Python or npm dependency.
- The app never writes `.env`; the model choice lives in `model_settings.json` (gitignored).
- `MODEL` stays the local base model (router, summaries, memory extraction).
- Claude Code runs with `--setting-sources project,local --strict-mcp-config` (no user hooks, plugins or MCP servers).
- Approval tools: `Edit`, `Write`, `MultiEdit`, `NotebookEdit`, `Bash`. Everything else Claude Code asks about is allowed.
- Claude aliases: `opus`, `sonnet`, `haiku`, `fable`. Efforts: `low`, `medium`, `high`, `xhigh`, `max`; default `medium`.
- Model id format: Ollama name as is (`qwen3:8b`), Claude as `claude-code:<alias>`.
- No `PROTOCOL` bump: every backend/frontend addition is optional for an old page.
- Pyright clean on every file, tests included (`tests/test_typecheck.py`).
- Python tests: `python -m unittest discover -s tests -t .`; frontend: `npm --prefix frontend run check`, `npm --prefix frontend test`, `npm --prefix frontend run build`.
- Per task: commit, push `feature/F036-cloud-models`, mark the step done in `feature-lists.json` F036 with tests and commit (new-feature skill).

## Review Focus

1. `claude` missing, not logged in, or plan limit hit: the answer says so in one plain sentence, the session keeps its model and the next turn works (Task 3, Task 5 tests).
2. Stop during a Claude turn, including while an approval is waiting: the `claude` process is killed (no orphan), the approval is denied, the next Claude turn resumes the same Claude session (Task 3, Task 5).
3. Vietnamese text in the prompt and the answer: sent and read as UTF-8 on Windows, not the console code page (Task 3).
4. `model_settings.json` missing, unreadable, or naming a model that is gone: startup warns and uses `MODEL`; never crashes (Task 1).
5. Ollama not running when the list is asked for: the last saved list plus `MODEL` is shown, Claude entries still listed (Task 1).

---

### Task 1: Model choice, live list, `model_settings.json`

**Files:**
- Create: `model_choice.py`
- Create: `tests/test_model_choice.py` (replaces the untracked `tests/test_providers.py`, which tests the dropped API-key design; delete it)
- Modify: `.gitignore` (add `model_settings.json`)

**Interfaces:**
- Produces:
  - `CLAUDE_PREFIX = "claude-code:"`, `CLAUDE_ALIASES: tuple[str, ...]`, `EFFORTS: tuple[str, ...]`, `DEFAULT_EFFORT = "medium"`, `SETTINGS_FILE = Path("model_settings.json")`
  - `@dataclass(frozen=True) class ModelChoice: model: str; effort: str | None` with properties `is_claude: bool`, `alias: str | None` and `label() -> str` (`"claude-code:sonnet (medium)"` / `"qwen3:8b"`)
  - `parse_choice(text: str) -> ModelChoice` — `"claude-code:opus high"`, `"claude-code:opus"` (effort `medium`), `"qwen3:4b"`; a bare alias `"opus"` also means Claude. Raises `ValueError` naming the usage `/model <name> [effort]` for an unknown alias or effort, or an effort on an Ollama model.
  - `ollama_models(env=None, fetch: Callable[[str], bytes] | None = None) -> list[str]` — GET `{OLLAMA_HOST or http://127.0.0.1:11434}/api/tags`, drop names containing `embed`, newest `modified_at` first. Raises `OSError` when unreachable.
  - `available_models(env=None, fetch=None, path: Path = SETTINGS_FILE) -> list[str]` — live Ollama list (saved to the settings file as `"ollama_models"`) or, on `OSError`, the saved list plus `MODEL`; then the four Claude ids.
  - `load_settings(path: Path = SETTINGS_FILE) -> dict` (missing or bad JSON → `{}` and a logged warning), `save_settings(data: dict, path: Path = SETTINGS_FILE) -> None` (write temp file, `os.replace`).
  - `default_choice(env=None, path: Path = SETTINGS_FILE) -> ModelChoice` — `"chat_model"`/`"effort"` from the file when they parse, else `ModelChoice(MODEL, None)` with a warning.
  - `save_default(choice: ModelChoice, path: Path = SETTINGS_FILE) -> None` — keeps other keys.
  - `privacy_notice(choice: ModelChoice) -> str | None` — a line for Claude choices only, e.g. "Claude Code runs this chat with your Claude login: your messages, long-term memory and the files it reads go to Anthropic."

- [ ] **Step 1: Write the failing tests** in `tests/test_model_choice.py`:
  - `test_parse_choice`: `parse_choice("claude-code:opus high") == ModelChoice("claude-code:opus", "high")`; `parse_choice("sonnet") == ModelChoice("claude-code:sonnet", "medium")`; `parse_choice("qwen3:4b") == ModelChoice("qwen3:4b", None)`; `ValueError` for `"claude-code:gpt"`, `"claude-code:opus extreme"`, `"qwen3:4b high"`.
  - `test_ollama_list_hides_embeddings_newest_first`: fake fetch returning `{"models":[{"name":"nomic-embed-text","modified_at":"2026-11-02T00:00:00Z"},{"name":"qwen3:4b","modified_at":"2026-11-01T00:00:00Z"},{"name":"qwen3:8b","modified_at":"2026-11-03T00:00:00Z"}]}` → `["qwen3:8b","qwen3:4b"]`.
  - `test_list_falls_back_when_ollama_is_down` (Review Focus 5): fetch raises `OSError`, settings file holds `{"ollama_models":["qwen3:4b"]}`, env `MODEL=qwen3:8b` → `["qwen3:4b","qwen3:8b","claude-code:opus","claude-code:sonnet","claude-code:haiku","claude-code:fable"]`.
  - `test_default_choice_falls_back_to_model` (Review Focus 4): missing file, file with `not json`, file with `{"chat_model":"claude-code:gpt"}` → each `ModelChoice("qwen3:8b", None)`, warning logged (`assertLogs`).
  - `test_save_default_keeps_other_keys`: file `{"ollama_models":["a"]}`, `save_default(ModelChoice("claude-code:opus","high"))` → file has all three keys.
  - `test_privacy_notice_only_for_claude`.
  All tests use a temp directory path, never the real `model_settings.json`.
- [ ] **Step 2:** Run `python -m unittest tests.test_model_choice -v`. Expected: FAIL, `ModuleNotFoundError: model_choice`.
- [ ] **Step 3:** Implement `model_choice.py` per the Interfaces block; `fetch` defaults to `urllib.request.urlopen(url, timeout=3).read()`. Delete `tests/test_providers.py`. Add `model_settings.json` to `.gitignore`.
- [ ] **Step 4:** Run `python -m unittest tests.test_model_choice -v` and `.venv\Scripts\pyright model_choice.py tests\test_model_choice.py`. Expected: PASS, 0 errors.
- [ ] **Step 5:** Commit `feat(F036): model choice, live Ollama list, model_settings.json`, push, update `feature-lists.json` step 1.

### Task 2: Probe the Claude Code permission channel (decision gate)

**Files:**
- Create (scratchpad, not committed): `probe_claude.py`
- Create: `tests/fixtures/claude_stream/text.jsonl`, `tool_read.jsonl`, `permission.jsonl`, `error.jsonl` (recorded stdout lines, trimmed to the fields Task 3 reads; no personal paths or account data)
- Modify: `docs/PROJECT.md` (Verified findings: the exact control-message shapes)

**Interfaces:**
- Produces: the recorded fixtures and the verified shapes of (a) partial text deltas, (b) assistant `tool_use` and user `tool_result` messages, (c) the permission request on stdout and the answer it accepts on stdin, (d) the final `result` message (`is_error`, `result`, `session_id`).

- [ ] **Step 1:** In the scratchpad, run `claude -p --input-format stream-json --output-format stream-json --verbose --include-partial-messages --model haiku --effort low --setting-sources project,local --strict-mcp-config --permission-prompts host --session-id <new uuid>` with `cwd` = a scratch folder holding one text file, writing one user message line to stdin (shape [assumed]: `{"type":"user","message":{"role":"user","content":"..."}}`). Ask it to read the file, then to create a second file. Record every stdout line.
- [ ] **Step 2:** Answer the permission request for the write once with allow and once (new run) with deny, using the [assumed] Agent SDK shape: request `{"type":"control_request","request_id":...,"request":{"subtype":"can_use_tool","tool_name":...,"input":{...}}}`, answer `{"type":"control_response","response":{"subtype":"success","request_id":...,"response":{"behavior":"allow","updatedInput":{...}}}}` or `{"behavior":"deny","message":"..."}`. Check the file was created only on allow.
- [ ] **Step 3:** Run once with `--resume <same uuid>` and ask "what file did you read?" — expect it to remember. Run once with `--model nosuchmodel` and once with a missing executable name to record error output.
- [ ] **Step 4:** Check that no user hook ran (the user's "Fact-Forcing Gate" PreToolUse hook would print its message on a Read/Write).
- [ ] **Step 5: Decision gate.** If allow/deny works as recorded: write the fixtures and the PROJECT.md finding, continue with Task 3 as written. If it does not: stop and tell the user; the agreed fallback is `--permission-prompts none` with a read-only permission mode and a per-session "allow edits" toggle (`acceptEdits` plus `Bash` allowed), which changes Task 3's `approve` handling and the Task 6/7/8 UI.
- [ ] **Step 6:** Commit fixtures + PROJECT.md `test(F036): record Claude Code stream-json shapes`, push, update step 2.

### Task 3: `claude_bridge.py`

**Files:**
- Create: `claude_bridge.py`
- Test: `tests/test_claude_bridge.py`

**Interfaces:**
- Consumes: `ModelChoice` (Task 1), fixtures (Task 2).
- Produces:
  - `APPROVAL_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit", "Bash"})`
  - `class ClaudeCodeError(RuntimeError)`
  - `session_uuid(thread_id: str) -> str` — `str(uuid.uuid5(uuid.NAMESPACE_URL, "langchain-agent/" + thread_id))`
  - `claude_command(alias: str, effort: str, session_id: str, resume: bool, exe: str = "claude") -> list[str]` — flags exactly as in Task 2 Step 1, `--resume <id>` when `resume` else `--session-id <id>`.
  - Events (frozen dataclasses): `Text(text: str)`, `ToolStart(id: str, name: str, args: dict)`, `ToolResult(id: str, text: str, is_error: bool)`, `Permission(request_id: str, tool: str, args: dict)`, `Done(text: str, is_error: bool)`
  - `parse_line(line: str) -> list[Event]` — unknown or non-JSON lines → `[]`.
  - `run_claude(command: list[str], prompt: str, cwd: str, approve: Callable[[dict], bool], cancel: threading.Event, popen=subprocess.Popen) -> Iterator[Event]` — starts the process with `encoding="utf-8"`, `errors="replace"`, stdin/stdout/stderr pipes; writes the user line; yields events; answers each `Permission` itself (tool not in `APPROVAL_TOOLS` → allow; else `approve({"calls": [{"name": tool, "args": args}]})`) and does not yield it; on `cancel` kills the process tree (`taskkill /T /F /PID` on Windows, `kill()` elsewhere) and raises `turns.TurnCancelled`; `FileNotFoundError` → `ClaudeCodeError("Claude Code is not installed: the claude command was not found.")`; a `Done(is_error=True)` or a non-zero exit with no `Done` → `ClaudeCodeError` carrying Claude's text (or the stderr tail).
  - `class ClaudeCodeChat(BaseChatModel)` — fields `alias: str`, `effort: str`, `cwd: str`, `session_id: str`, `resume: bool`, `approve: Callable[[dict], bool]`, `cancel: threading.Event`, `runner: Callable[..., Iterator[Event]] = run_claude` (pydantic `arbitrary_types_allowed`). `_stream` yields `AIMessageChunk(content=text)` for `Text`, and `AIMessageChunk(content="", additional_kwargs={"claude_tool": {"id", "name", "args", "status": "running" | "done" | "error", "text"}})` for tool events. Uses only the last human message as the prompt. `_generate` joins the stream.

- [ ] **Step 1: Write the failing tests:**
  - `test_command_flags`: `claude_command("sonnet","medium","U",False)` contains `["--model","sonnet"]`, `["--effort","medium"]`, `["--setting-sources","project,local"]`, `"--strict-mcp-config"`, `["--permission-prompts","host"]`, `["--session-id","U"]` and not `"--resume"`; with `resume=True` the reverse.
  - `test_parse_fixtures`: each fixture parses to the expected event list (text pieces join to the recorded answer; `tool_read.jsonl` gives one `ToolStart(name="Read")` and one `ToolResult`; `permission.jsonl` one `Permission(tool="Write")`; `error.jsonl` ends with `Done(is_error=True)`).
  - `test_read_tools_are_allowed_without_asking` and `test_edit_asks_and_deny_is_sent`: fake `popen` whose stdout replays `permission.jsonl`; assert `approve` is called once with `{"calls":[{"name":"Write","args":{...}}]}` and the line written to stdin is the deny answer when `approve` returns False.
  - `test_cancel_kills_and_raises` (Review Focus 2): fake process blocking on stdout; set `cancel` from another thread; assert kill called and `TurnCancelled` raised within 2 s; also when `cancel` is set while `approve` is waiting (approve returns False after cancel).
  - `test_missing_claude_is_a_clear_error` (Review Focus 1): `popen` raising `FileNotFoundError` → `ClaudeCodeError` with "not installed".
  - `test_error_result_is_raised` (Review Focus 1): replay `error.jsonl` → `ClaudeCodeError` whose text is Claude's message.
  - `test_vietnamese_round_trip` (Review Focus 3): fake process records stdin; prompt "Sửa lỗi trong file này" written unchanged; a recorded text line with "Đã sửa" parses back unchanged; assert `popen` got `encoding="utf-8"`.
  - `test_chat_model_streams_text_and_tool_chunks`: `ClaudeCodeChat` with a fake `runner` yielding `Text("Hi")`, `ToolStart(...)`, `ToolResult(...)`, `Text(" there")` → `.stream()` chunk contents `"Hi"`, `""`, `""`, `" there"`; the two tool chunks carry `claude_tool` with status `running`, `done`.
- [ ] **Step 2:** Run `python -m unittest tests.test_claude_bridge -v`. Expected: FAIL, module missing.
- [ ] **Step 3:** Implement `claude_bridge.py` per Interfaces. Message shapes come from the Task 2 fixtures, not from memory.
- [ ] **Step 4:** Run the tests and Pyright on both files. Expected: PASS, 0 errors.
- [ ] **Step 5:** Commit `feat(F036): Claude Code bridge (stream-json, approvals, cancel)`, push, update steps 3 and 4 (this task covers approvals).

### Task 4: Per-turn model in the graph (Ollama switching)

**Files:**
- Modify: `turns.py` (`run_turn`, `_run_turn`), `graph/node.py` (`direct`, `code_tools`)
- Create: `chains/by_model.py`
- Test: `tests/test_model_switching.py`

**Interfaces:**
- Consumes: `ModelChoice` (Task 1).
- Produces:
  - `run_turn(..., model: ModelChoice | None = None)` — copies `config` and sets `configurable["model"] = model`, `configurable["approve"] = approve`, `configurable["cancel"] = cancel`; `None` means `MODEL` (old callers unchanged).
  - `chains.by_model.chains_for(name: str) -> tuple[Runnable, Runnable]` — `(direct_chain, code_chain)` for that Ollama model; `MODEL` returns the existing chains; others are built once (`ChatOllama(model=name, **ollama_options())`, same prompts, `bind_tools(TOOLS)`) and cached.
  - `graph.node.turn_model(config: RunnableConfig) -> ModelChoice | None`

- [ ] **Step 1: Write the failing tests:** with `chains_for` patched to fakes, a turn with `model=ModelChoice("qwen3:4b", None)` calls the `qwen3:4b` fake in `direct`; a turn with no model calls the `MODEL` chains; `chains_for("qwen3:4b")` returns the same objects on the second call; `run_turn` passes `approve` and `cancel` into `configurable` (fake app records the config).
- [ ] **Step 2:** Run `python -m unittest tests.test_model_switching -v`. Expected: FAIL.
- [ ] **Step 3:** Implement. `direct`/`code_tools` take `config: RunnableConfig` and pick chains via `chains_for(turn_model(config).model)` for an Ollama choice.
- [ ] **Step 4:** Run the new tests plus `tests.test_graph`, `tests.test_chat_loop`, `tests.test_tool_activity`. Expected: PASS.
- [ ] **Step 5:** Commit `feat(F036): per-turn Ollama model`, push, note it under step 5 in `feature-lists.json`.

### Task 5: `claude_code` graph node

**Files:**
- Modify: `graph/node.py` (`route`, `chosen_route`, new `claude_code`, `claude_prompt`), `graph/graph.py`, `data/state.py` (add `claude_session: str | None`), `turns.py` (`STREAMED_NODES`, `SPOKEN_NODES`, `ToolActivity.feed`)
- Test: `tests/test_claude_node.py`

**Interfaces:**
- Consumes: `ClaudeCodeChat`, `ClaudeCodeError`, `session_uuid` (Task 3); `turn_model` and the `configurable` keys (Task 4).
- Produces:
  - `route(state, config)` returns `{"route": "claude_code"}` without calling the router when `turn_model(config).is_claude`.
  - `claude_prompt(messages: list, memory_context: str, resumed: bool) -> str` — memory first ("Long-term memory (reference, not a request): …"), then, when there are messages after the last Claude Code answer (or no Claude answer yet and `resumed` is False), a recap "Earlier in this chat (you did not see it):" of the last 12 such messages, each cut to 500 chars, then the user's message. Notices and tool messages are left out of the recap.
  - `claude_code(state, config) -> dict` — runs `ClaudeCodeChat(alias, effort, cwd=PROJECT_ROOT, session_id=session_uuid(thread_id), resume=state.get("claude_session") == that id, approve, cancel)` via `.invoke` with one `HumanMessage(claude_prompt(...))`; returns `{"messages": [AIMessage(answer, additional_kwargs={"model": choice.model})], "claude_session": id}`. On `ClaudeCodeError` the answer is `"Claude Code could not answer: <text>"` and `claude_session` is unchanged.
  - Graph: `"claude_code"` in the `routed` conditional map; `claude_code → save_memory`.
  - `ToolActivity.feed("claude_code", chunk)` reports `chunk.additional_kwargs["claude_tool"]` and returns True; text chunks return False. `claude_code` joins `STREAMED_NODES` and `SPOKEN_NODES` (held like `code_tools`).

- [ ] **Step 1: Write the failing tests** (fake `ClaudeCodeChat` factory patched in `graph.node`):
  - a turn with a Claude choice never calls `router_chain`; the answer streams to `on_text`; `save_memory` runs; the stored AI message has `additional_kwargs["model"]`.
  - `claude_prompt`: after two Ollama turns the first Claude turn's prompt holds both as recap; the next Claude turn (resumed, nothing new) holds no recap; memory text comes first.
  - the second Claude turn in a thread is built with `resume=True`; after a new thread id `resume=False`.
  - `ClaudeCodeError("not logged in")` → answer "Claude Code could not answer: not logged in", next turn still runs (Review Focus 1).
  - `TurnCancelled` from the fake propagates out of `run_turn` and the next turn runs (Review Focus 2).
  - `on_tool` receives running/done reports from `claude_tool` chunks and their text does not reach `on_text`.
- [ ] **Step 2:** Run `python -m unittest tests.test_claude_node -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the full suite. Expected: all pass (Postgres tests skip if Docker is down; use `DB_URL=...?connect_timeout=3` per PROJECT.md).
- [ ] **Step 5:** Commit `feat(F036): Claude Code turns in the graph`, push, update step 5.

### Task 6: CLI `/model`

**Files:**
- Modify: `chat_loop.py` (command handling beside `/jobs`; `_run` passes `model=self.choice`), `main.py` (initial choice)
- Test: `tests/test_chat_loop.py`

**Interfaces:**
- Consumes: `available_models`, `parse_choice`, `default_choice`, `save_default`, `privacy_notice` (Task 1); `run_turn(model=...)` (Task 4).
- Produces: `ChatLoop.choice: ModelChoice`; `/model` prints the current choice and the numbered list; `/model <name|number> [effort]` sets it, saves the default, prints the privacy line for Claude; works while a turn runs and applies from the next turn; invalid input prints the `ValueError` text.

- [ ] **Step 1: Write the failing tests:** `/model` lists `available_models` (patched); `/model claude-code:opus high` sets `loop.choice`, calls `save_default` (patched) and prints the privacy line; the next turn's `run_turn` gets `model=ModelChoice("claude-code:opus","high")`; `/model nope` prints the usage and keeps the choice.
- [ ] **Step 2:** Run `python -m unittest tests.test_chat_loop -v`. Expected: new tests FAIL.
- [ ] **Step 3:** Implement; `main.py` builds `ChatLoop` with `default_choice()` and prints the privacy line at start for a Claude choice.
- [ ] **Step 4:** Run `tests.test_chat_loop`, `tests.test_streaming_speech`, `tests.test_vietnamese`. Expected: PASS.
- [ ] **Step 5:** Commit `feat(F036): /model in the CLI`, push, update step 6.

### Task 7: Backend per-session model

**Files:**
- Modify: `backend/session.py` (`ChatSession.choice`, `set_model`, `info`, `_turn`), `backend/server.py` (routes, `describe`)
- Test: `tests/test_backend_session.py`, `tests/test_backend_handshake.py`, new `tests/test_backend_models.py`

**Interfaces:**
- Consumes: Task 1 functions; `run_turn(model=...)`.
- Produces:
  - `ChatSession.choice: ModelChoice` (starts at `default_choice()`); `ChatSession.set_model(choice: ModelChoice) -> bool` (False while busy); emits `{"type": "model", "model": ..., "effort": ..., "text": <privacy line or "">}`; `info()` adds `"model"` and `"effort"`.
  - `GET /api/models` → `{"models": [...], "efforts": [...], "default": {"model", "effort"}}`.
  - `POST /api/sessions/<id>/model` body `{"model": str, "effort": str | null}` → 200 session info; 400 with the `ValueError` text; 409 while busy; 404 unknown session. Same token/CORS rules as the other session routes.
  - `describe()["features"]["models"] = True`.

- [ ] **Step 1: Write the failing tests:** route answers above (200/400/409/404, token required); a turn after `set_model` passes the choice to `run_turn`; the `model` event carries the privacy text for Claude; handshake lists `features.models`.
- [ ] **Step 2:** Run the three test modules. Expected: new tests FAIL.
- [ ] **Step 3:** Implement. A page pick does not call `save_default`.
- [ ] **Step 4:** Run all `tests.test_backend_*`. Expected: PASS.
- [ ] **Step 5:** Commit `feat(F036): per-session model in the web API`, push, update step 7.

### Task 8: Frontend picker

**Files:**
- Modify: `frontend/src/types.ts` (`SessionInfo.model?`, `SessionInfo.effort?`, `ServerInfo.features.models?`, event `model`), `frontend/src/api.ts` (`listModels()`, `setModel(id, model, effort)`), `frontend/src/commands.ts` (`/model`), `frontend/src/main.ts` (header picker, palette entries), `frontend/src/state.ts` (render `model` events)
- Test: `frontend/test/api.test.ts`, `frontend/test/commands.test.ts`, `frontend/test/state.test.ts`

**Interfaces:**
- Consumes: Task 7 routes and event.
- Produces: a model `<select>` plus effort `<select>` (shown for Claude ids only) in the session header, hidden when `features.models` is absent; `/model <name> [effort]` slash command (exact-name rule as the others); palette entries "Model: <id>" per listed model; a `model` event renders as a system line with the privacy text.

- [ ] **Step 1: Write the failing tests:** `api.setModel` posts `{model, effort}` to `/api/sessions/<id>/model`; the `/model claude-code:opus high` slash command parses to command `model` with its argument; reducing a `model` event adds one system line holding the text; an unknown event is still dropped.
- [ ] **Step 2:** Run `npm --prefix frontend test`. Expected: new tests FAIL.
- [ ] **Step 3:** Implement. No ES2023 array methods (PROJECT.md: the `tsc` lib lacks them).
- [ ] **Step 4:** Run `npm --prefix frontend run check`, `npm --prefix frontend test`, `npm --prefix frontend run build`. Expected: all pass.
- [ ] **Step 5:** Commit `feat(F036): model picker in the web chat`, push, update step 8.

### Task 9: Docs and real run

**Files:**
- Modify: `CLAUDE.md` (Architecture: model choice, `claude_code` node, bridge; Commands: `/model`), `backend/CLAUDE.md`, `frontend/CLAUDE.md`, `README.md`, `.env.example` (comment: Claude runs through Claude Code, no key), `docs/PROJECT.md` (Current state; Decisions entry for F036 with the rejected API-key path), `feature-lists.json` (F036 done)

- [ ] **Step 1:** Real run: `python main.py`, `/model claude-code:haiku low`, ask it to read `README.md` and summarize (no approval asked), then to create `scratch.txt` (approval asked; answer n; file not created), then Stop mid-answer, then one more question (resumes). Record the result in PROJECT.md as [verified], or the failure found.
- [ ] **Step 2:** Same in the web chat (`python -m backend`, frontend): approval card shown for the write, tool cards for the read.
- [ ] **Step 3:** Update the docs listed above; run the full Python suite, Pyright and frontend checks.
- [ ] **Step 4:** Commit `docs(F036): Claude Code bridge`, push, mark F036 done; open the PR to `main` when the user asks.
