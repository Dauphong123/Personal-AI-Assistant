# CLAUDE.md

Local LLM agent: LangChain + LangGraph on Ollama (default `qwen3:8b`). Interactive CLI chat; each query is routed to a plain chat answer or a coding agent (filesystem/shell/git tools). Chroma RAG over `./documents/`; Postgres/pgvector long-term memory (see `memories/CLAUDE.md`); optional voice mode (see `voice/CLAUDE.md`).

## Commands

```powershell
.\.venv\Scripts\Activate.ps1
python main.py                              # interactive chat; "exit" or "quit" stops
npm --prefix frontend install; npm --prefix frontend run build   # once: build the TypeScript web frontend
python -m backend [--no-voice]              # agent web API on 127.0.0.1:8000: many sessions, job notices, voice if installed; prints the frontend link (api + token)
npm --prefix frontend start                 # web frontend on 127.0.0.1:5173, calls the API (python -m backend --serve-frontend = both on :8000 instead)
python -m unittest discover -s tests -t .   # tests (Postgres test skipped if DB unreachable)
docker compose up -d                        # pgvector Postgres 17 (db agent_memory, user agent)
```

CI: `.github/workflows/ci.yml` runs the unittest suite (Windows, `.venv` with `requirements.txt` + `requirements_voice.txt` locked by `constraints.txt` (`pip freeze`; regenerate after changing requirements), pyright, ripgrep, built `frontend/`) and the `webui` job (check/test/build of `frontend/`; job id kept because it is a required check) on every push and on PRs to `main`; no Ollama/Postgres/voice engines there. `main` is protected: `python` and `webui` must pass, linear history, no force push (fast-forward a green branch or merge the PR). Keep it in step with these commands (`tests/test_ci_workflow.py`).

Voice commands (`--voice`, `voice.enroll`, `voice.check`, voice engine tests): `voice/CLAUDE.md`.

Python 3.12, Windows, venv `.venv`, deps in `requirements.txt`. Needs Ollama with `qwen3:8b` (main; summaries use it too unless `SUMMARY_MODEL` names another model), and `rg` (ripgrep) for `search_code`. First run downloads the `Qwen/Qwen3-8B` tokenizer, `BAAI/bge-small-en-v1.5` (RAG) and `intfloat/multilingual-e5-small` (memory) embeddings.

`.env` (gitignored; copy `.env.example`): `MODEL` must be a key in `TOKENIZER_MAP` (`llm_options.py`; `model_name()` fails at startup with a clear message otherwise). DB URL from `DATABASE_URL` or `DB_URL`; `postgresql://` is rewritten to `postgresql+psycopg://` (only psycopg 3 installed). `PROJECT_ROOT` changes the directory the coding tools may touch.

**Speed.** `llm_options.py` sets `reasoning=False` on every `ChatOllama` (thinking made routing take 113 s vs 3 s), `keep_alive=30m`, optional `LLM_NUM_CTX`. `MAX_HISTORY_TOKENS` (default 5000) must stay below the context window (8192) minus the ~1,400-token tool prompt and the answer, or Ollama silently drops the oldest messages. This machine (RTX 3050 4 GB): `qwen3:8b` ~4-6 tok/s, ~64% CPU, so output length is the main cost; only one model fits (a second one unloads it, ~8 s reload), so summaries use the main model; `qwen3:4b` still thinks with tools (`tests/test_model_eval.py`, opt-in). Ollama reuses work only for an unchanged prompt start and its qwen3 template puts every system message at the top, so memory goes in a user-role message before the latest one (`with_memory`). `LOG_TIMING=true` logs node durations and token counts.

**Port gotcha.** A native Windows `postgres` also listens on `localhost:5432`, so connections hit the wrong server ("password authentication failed" while the container is fine). `compose.yaml` therefore publishes the container on host port **15432** (native servers also own 5433, and Windows lets them bind beside Docker; 50000-55633 are reserved by Windows); `DB_URL` must use `localhost:15432`. `python -m memories.check` diagnoses a failing connection (names the port owner, prints the fix); the memory startup warning includes the same hint.

## Architecture

**Import-time singleton.** `dependency.py` builds model, RAG, tokenizer and memory manager on import (`dependency = dependency_init()`); `chains/*`, `tools.py`, `graph/node.py` use `from dependency import dependency`. Importing nearly any module loads models and syncs `./documents/` into Chroma (idempotent: stable chunk ids, stale chunks deleted). `dependency.py` must not import `graph` (circular). `main.py` compiles the graph.

**Graph** (`graph/graph.py`, `graph/node.py`, state `data/state.py` = `{messages (add_messages), memory_context, route}`):

- `START → recall_memory ∥ route → routed → direct | code_tools | claude_code`; `route` uses `quick_route` rules for obvious messages, else the router model (sees the last few messages); a Claude Code turn skips the router.
- **Model choice (F036).** `model_choice.py`: ids are an Ollama name (`qwen3:4b`) or `claude-code:<opus|sonnet|haiku|fable>` with an effort (`low`…`max`, default `medium`); live Ollama list (`/api/tags`, embeddings hidden, last list kept for when Ollama is down); the pick lives in `model_settings.json` (gitignored; the app never writes `.env`). `run_turn(model=...)` puts the choice plus `approve`/`cancel` in `configurable`; `direct`/`code_tools` use `model_chains.chains_for(name)` for an Ollama model other than `MODEL` (router, summaries, memory stay on `MODEL`). `claude_bridge.py` runs the user's own `claude -p` (stream-json, `--setting-sources project,local --strict-mcp-config`, `--permission-prompts host`) in `PROJECT_ROOT`, one Claude session per thread (`--session-id`/`--resume`, `State.claude_session`); `Edit`/`Write`/`MultiEdit`/`NotebookEdit`/`Bash` go to our `approve`, other tools are allowed; its tools arrive as `claude_tool` chunks (`ToolActivity`). `graph.node.claude_prompt`: memory, then a recap of talk Claude's session missed, then the message. Stream-json shapes are [assumed] (`tests/fixtures/claude_stream`) until a real run checks them.
- `code_tools ⇄ tool_execute` while the AI message has `tool_calls`; then `→ save_memory`. `direct → save_memory`. Each model call's messages go through `fit_to_context` (older tool results trimmed; budget = context − 1,400 − 1,500).
- History over `MAX_HISTORY_TOKENS` (counts text and tool-call args): `summarize` condenses everything before the latest turn; the latest turn stays whole (tool calls never split from results).
- `tool_execute` never raises: unknown tools/errors become `ERROR:` tool messages, output capped, max 30 tool calls per turn. Arguments go through `tool_args.prepare` first: a dict/list for a `str` parameter becomes JSON text again (langchain_ollama parses JSON-looking strings), a synonym (`file_path`, `dir`/`directory`/`repo_path`, `old_string`, ...) is renamed when exactly one parameter matches, unknown/invalid arguments are refused with the tool's signature, before the approval prompt. Tools in `NEEDS_APPROVAL` (`tools.py`: terminal, write/edit/delete/move) pause via `interrupt()`; `ChatLoop` asks y/N (voice asks aloud) and resumes with `Command(resume=bool)`.
- `main.py` uses `InMemorySaver` (thread_id `"1"`); `turns.py` streams chunks from `direct`, `code_tools`, `tool_execute`. With `run_turn(on_tool=...)` (the web backend) tool calls are reported apart (`turns.ToolActivity`: start + result) and tool output leaves the answer text; the CLI passes none and still prints it.

**Live input / jobs.** `main.py` only reads the keyboard; `chat_loop.py` (`ChatLoop`) runs turns one at a time on a worker thread. Mid-turn messages are queued; a short stop message (`is_stop_request` in `turns.py`: "stop", "cancel that", `/stop`, Vietnamese phrases) cancels the turn and drops the queue (job notices kept). Approval answers use the same input line. `/jobs`, `/cancel <id>`, `/model [name|number] [effort]` (lists / switches from the next turn and saves the default) work any time. `turns.py` holds UI-independent `run_turn`; a cancelled turn's unanswered tool calls get `ERROR: cancelled` before the next turn. `terminal` runs via `jobs.py` (`JobManager`, global `tools.JOBS`): past `FOREGROUND_WAIT` (5 s) it goes to background, the tool returns a job id, completion queues a `[system notice]` turn (text from `jobs.finished_notice`; jobs carry `owner` = `jobs.JOB_OWNER`, which the web backend uses to route the notice to its session); `job_status` / `cancel_job` tools; jobs killed after 10 min.

**Streaming speech.** `turns.SpeechStream` turns streamed answers (nodes `direct`, `code_tools` only) into sentences: tool-calling messages muted, `code_tools` holds its newest sentence until the next starts (tool call arrives after narration), code blocks become "(code omitted)", stops at `speech_limit`. `run_turn(on_speech=...)` returns whether anything streamed; then `ChatLoop` skips `on_reply` (no double speech). `ChatLoop.on_cancel` drops queued speech.

**Search.** `retrieval.py`: shared BM25 (stop words, common-word filter, accent folding), weighted reciprocal rank fusion, passage splitting. `RAG.search_hybrid` fuses vector + BM25, drops vector hits beyond `MAX_VECTOR_DISTANCE` (0.9; bge-small: related <= 0.70, unrelated >= 1.06), max 2 passages per file, empty for unrelated queries. `web.py`: `web_search` (ddgs, cached), `read_webpage` (SSRF-guarded: public addresses only, redirects re-checked, DNS pinned to checked addresses, size/time/type limits; `focus` = BM25 passages); output labelled untrusted; `WEB_SEARCH_ENABLED=false` disables both.

**Vietnamese.** `language.py` (`detect_language`, `say`, NFC `normalize`) is shared by `chat_loop.py`, `turns.py`, `voice/`. Prompts say to answer in the user's language. RAG embeddings are English-only (weak Vietnamese recall), so the code prompt says to search documents in English; memory embeddings are multilingual (`memories/CLAUDE.md`). Speech-language settings: `voice/CLAUDE.md`.

**Hardening.** Job notices reach the model as a flagged, fenced message (`additional_kwargs["notice"]`); memory neither recalls for nor learns from them. `ChatLoop.close()` cancels; `wait_idle()` lets piped input finish. `JobManager.close()` kills running jobs; job output is a bounded tail. `save_memory` queues on `memories/scheduler.py` (one worker; starts after `MEMORY_IDLE_SECONDS` with no turn; `run_turn` cancels a running save, which retries; `wait_for_memory()` at exit runs the queue). `read_file`/`search_code` refuse credential files (`SECRET_FILE_PATTERNS`).

**Settings and logs.** `dependency.py` runs `settings.check_settings` right after `load_dotenv` and exits with every bad value listed (MODEL, flags, numbers, `PROJECT_ROOT`, `LOG_LEVEL`); add new settings there. `log_setup.configure_logging` (both entry points): `LOG_LEVEL` for the console (default WARNING), `LOG_FILE` = rotating file at INFO; a crashing tool is logged at INFO. Pyright checks every file, tests included (`tests/test_typecheck.py`; fakes get typed or a file-level directive with its reason).

**Adding a tool:** `@tool` in `tools.py` + add to `TOOLS` at the bottom (binding and dispatch derive from it); add a line in the code prompt in `chains/code_chain.py` (a test checks every tool named there exists); add to `NEEDS_APPROVAL` if it changes state or runs commands; file args go through `resolve_path()` (rejects paths outside `PROJECT_ROOT`; `/x` without a drive means `PROJECT_ROOT/x`); reuse the existing parameter names or add the new name to `tool_args.SYNONYMS`.

**Naming quirk:** the summarization chain is `summerize` in file/symbol names (`summerize_chain`); graph nodes use `summarize`.

**Known leftovers:** `transformers` is used only to count tokens (`count_message_tokens`: the summarize trigger and `fit_to_context`).

## How to work here

Improve the existing project; do not rebuild it. Respect reasonable existing design, reuse existing code, preserve behavior, avoid needless abstraction, add dependencies only with clear justification. A different architecture is not automatically better.

- Inspect and trace the real source before proposing anything; never infer from file names. Never invent functions, classes or behavior; investigate when unsure.
- Report only issues you can substantiate. For a full review checklist use the `architecture-review` skill (`.claude/skills/architecture-review/SKILL.md`).
- Present a plan first; implement approved items incrementally, one unrelated change at a time. Per change: inspect files, explain current vs proposed, edit only necessary files, update imports/interfaces, add/update tests, run tests, inspect `git diff`, report changes and remaining issues.
- Ask first for breaking interface changes, major dependencies, or architecture changes. Don't delete working functionality or rewrite working components without a clear reason.
- Verify before claiming success: run relevant tests, check imports/initialization and affected paths, and say which tests could not run and why.
- Never modify unrelated files, overwrite config/`.env` without permission, expose secrets, or commit/push without permission.
- Tools: follow Superpowers workflows for non-trivial features; Serena for semantic navigation/refactoring; Pyright for type errors; code review before finalizing significant changes.

## Agent skills

### Issue tracker

GitHub Issues on `Dauphong123/Personal-AI-Assistant` via `gh` (not on PATH: use `C:\Program Files\GitHub CLI\gh.exe`); `feature-lists.json` stays the feature log. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` + `docs/adr/` at the root, created lazily. See `docs/agents/domain.md`.

## Workflow and shared context

Solo project: roles are skills (`architect`, `implement`, `test`, `debug`, `docs-sync`), loaded on demand. Read the roster and project knowledge below at session start; run `docs-sync` at session end so the next session resumes without rediscovery.

- Label claims verified (cite code/test/doc), assumed, or unknown. Investigate uncertainty; surface what stays unresolved to the user rather than guessing.
- Verify important findings against source, tests, or authoritative docs.
- Directory notes: `voice/CLAUDE.md`, `memories/CLAUDE.md`, `backend/CLAUDE.md` (web API, Python), `frontend/CLAUDE.md` (web frontend, TypeScript). The two web parts meet only at the HTTP API: the frontend's own server (`frontend/scripts/serve.ts`, :5173) serves the page; the page calls the backend (:8000) cross-origin (CORS for `--frontend-origin`, token in `X-Token`).

@docs/ROSTER.md
@docs/PROJECT.md
