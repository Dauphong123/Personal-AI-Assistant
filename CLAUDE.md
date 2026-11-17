# CLAUDE.md

Local LLM agent: LangChain + LangGraph on Ollama (default `qwen3:8b`). Interactive CLI chat; each query is routed to a plain chat answer or a coding agent (filesystem/shell/git tools). Chroma RAG over `./documents/`; Postgres/pgvector long-term memory (see `memories/CLAUDE.md`); optional voice mode (see `voice/CLAUDE.md`).

## Commands

```powershell
.\.venv\Scripts\Activate.ps1
python main.py                              # interactive chat; "exit" or "quit" stops
npm --prefix webui install; npm --prefix webui run build   # once: build the TypeScript web frontend
python -m backend                           # browser chat on 127.0.0.1:8000, many sessions (link with token is printed)
python -m unittest discover -s tests -t .   # tests (Postgres test skipped if DB unreachable)
docker compose up -d                        # pgvector Postgres 17 (db agent_memory, user agent)
```

Voice commands (`--voice`, `voice.enroll`, `voice.check`, voice engine tests): `voice/CLAUDE.md`.

Python 3.12, Windows, venv `.venv`, deps in `requirements.txt`. Needs Ollama with `qwen3:8b` (main) and `qwen3:4b` (summaries; `SUMMARY_MODEL`), and `rg` (ripgrep) for `search_code`. First run downloads the `Qwen/Qwen3-8B` tokenizer and `BAAI/bge-small-en-v1.5` embeddings.

`.env` (gitignored; copy `.env.example`): `MODEL` must be a key in `TOKENIZER_MAP` (`llm_options.py`; `model_name()` fails at startup with a clear message otherwise). DB URL from `DATABASE_URL` or `DB_URL`; `postgresql://` is rewritten to `postgresql+psycopg://` (only psycopg 3 installed). `PROJECT_ROOT` changes the directory the coding tools may touch.

**Speed.** `llm_options.py` sets `reasoning=False` on every `ChatOllama` (thinking made routing take 113 s vs 3 s), `keep_alive=30m`, optional `LLM_NUM_CTX`. `MAX_HISTORY_TOKENS` (default 5000) must stay below the context window (8192) minus the ~1,400-token tool prompt and the answer, or Ollama silently drops the oldest messages. This machine (RTX 3050 4 GB): `qwen3:8b` ~4 tok/s, ~64% CPU, so output length is the main cost; `qwen3:4b` still thinks.

**Port gotcha.** Another process also answers on `localhost:5432`, so connections can hit the wrong server ("password authentication failed" while the container is fine). If memory logs a startup warning, check the port owner or map the container to another host port and update `DB_URL`.

## Architecture

**Import-time singleton.** `dependency.py` builds model, RAG, tokenizer and memory manager on import (`dependency = dependency_init()`); `chains/*`, `tools.py`, `graph/node.py` use `from dependency import dependency`. Importing nearly any module loads models and syncs `./documents/` into Chroma (idempotent: stable chunk ids, stale chunks deleted). `dependency.py` must not import `graph` (circular). `main.py` compiles the graph.

**Graph** (`graph/graph.py`, `graph/node.py`, state `data/state.py` = `{messages (add_messages), search_file, memory_context}`):
- `START → recall_memory → router → direct | code_tools`; router sees the last few messages.
- `code_tools ⇄ tool_execute` while the AI message has `tool_calls`; then `→ save_memory → reverse_router`. `direct → save_memory`.
- History over `MAX_HISTORY_TOKENS` (counts text and tool-call args): `summarize` condenses everything before the latest turn; the latest turn stays whole (tool calls never split from results).
- `tool_execute` never raises: unknown tools/errors become `ERROR:` tool messages, output capped, max 30 tool calls per turn. Tools in `NEEDS_APPROVAL` (`tools.py`: terminal, write/edit/delete/move) pause via `interrupt()`; `ChatLoop` asks y/N (voice asks aloud) and resumes with `Command(resume=bool)`.
- `main.py` uses `InMemorySaver` (thread_id `"1"`); `turns.py` streams chunks from `direct`, `code_tools`, `tool_execute`.

**Live input / jobs.** `main.py` only reads the keyboard; `chat_loop.py` (`ChatLoop`) runs turns one at a time on a worker thread. Mid-turn messages are queued; a short stop message (`is_stop_request` in `turns.py`: "stop", "cancel that", `/stop`, Vietnamese phrases) cancels the turn and drops the queue (job notices kept). Approval answers use the same input line. `/jobs`, `/cancel <id>` work any time. `turns.py` holds UI-independent `run_turn`; a cancelled turn's unanswered tool calls get `ERROR: cancelled` before the next turn. `terminal` runs via `jobs.py` (`JobManager`, global `tools.JOBS`): past `FOREGROUND_WAIT` (5 s) it goes to background, the tool returns a job id, completion queues a `[system notice]` turn; `job_status` / `cancel_job` tools; jobs killed after 10 min.

**Streaming speech.** `turns.SpeechStream` turns streamed answers (nodes `direct`, `code_tools` only) into sentences: tool-calling messages muted, `code_tools` holds its newest sentence until the next starts (tool call arrives after narration), code blocks become "(code omitted)", stops at `speech_limit`. `run_turn(on_speech=...)` returns whether anything streamed; then `ChatLoop` skips `on_reply` (no double speech). `ChatLoop.on_cancel` drops queued speech.

**Search.** `retrieval.py`: shared BM25 (stop words, common-word filter, accent folding), weighted reciprocal rank fusion, passage splitting. `RAG.search_hybrid` fuses vector + BM25, drops vector hits beyond `MAX_VECTOR_DISTANCE` (0.9; bge-small: related <= 0.70, unrelated >= 1.06), max 2 passages per file, empty for unrelated queries. `web.py`: `web_search` (ddgs, cached), `read_webpage` (SSRF-guarded: public addresses only, redirects re-checked, DNS pinned to checked addresses, size/time/type limits; `focus` = BM25 passages); output labelled untrusted; `WEB_SEARCH_ENABLED=false` disables both.

**Vietnamese.** `language.py` (`detect_language`, `say`, NFC `normalize`) is shared by `chat_loop.py`, `turns.py`, `voice/`. Prompts say to answer in the user's language. Memory/RAG embeddings are English-only (weak Vietnamese recall), so the code prompt says to search documents in English. Speech-language settings: `voice/CLAUDE.md`.

**Hardening.** Job notices reach the model as a flagged, fenced message (`additional_kwargs["notice"]`); memory neither recalls for nor learns from them. `ChatLoop.close()` cancels; `wait_idle()` lets piped input finish. `JobManager.close()` kills running jobs; job output is a bounded tail. `save_memory` runs in a one-worker background pool (`wait_for_memory()` at exit). `read_file`/`search_code` refuse credential files (`SECRET_FILE_PATTERNS`).

**Adding a tool:** `@tool` in `tools.py` + add to `TOOLS` at the bottom (binding and dispatch derive from it); add a line in the code prompt in `chains/code_chain.py` (a test checks every tool named there exists); add to `NEEDS_APPROVAL` if it changes state or runs commands; file args go through `resolve_path()` (rejects paths outside `PROJECT_ROOT`).

**Naming quirk:** the summarization chain is `summerize` in file/symbol names (`summerize_chain`); graph nodes use `summarize`.

**Known leftovers:** `transformers` only counts tokens for the summarize trigger.

## How to work here

Improve the existing project; do not rebuild it. Respect reasonable existing design, reuse existing code, preserve behavior, avoid needless abstraction, add dependencies only with clear justification. A different architecture is not automatically better.

- Inspect and trace the real source before proposing anything; never infer from file names. Never invent functions, classes or behavior; investigate when unsure.
- Report only issues you can substantiate. For a full review checklist use the `architecture-review` skill (`.claude/skills/architecture-review/SKILL.md`).
- Present a plan first; implement approved items incrementally, one unrelated change at a time. Per change: inspect files, explain current vs proposed, edit only necessary files, update imports/interfaces, add/update tests, run tests, inspect `git diff`, report changes and remaining issues.
- Ask first for breaking interface changes, major dependencies, or architecture changes. Don't delete working functionality or rewrite working components without a clear reason.
- Verify before claiming success: run relevant tests, check imports/initialization and affected paths, and say which tests could not run and why.
- Never modify unrelated files, overwrite config/`.env` without permission, expose secrets, or commit/push without permission.
- Tools: follow Superpowers workflows for non-trivial features; Serena for semantic navigation/refactoring; Pyright for type errors; code review before finalizing significant changes.

## Workflow and shared context

Solo project: roles are skills (`architect`, `implement`, `test`, `debug`, `docs-sync`), loaded on demand. Read the roster and project knowledge below at session start; run `docs-sync` at session end so the next session resumes without rediscovery.

- Label claims verified (cite code/test/doc), assumed, or unknown. Investigate uncertainty; surface what stays unresolved to the user rather than guessing.
- Verify important findings against source, tests, or authoritative docs.
- Directory notes: `voice/CLAUDE.md`, `memories/CLAUDE.md`, `backend/CLAUDE.md` (web API, Python), `webui/CLAUDE.md` (web frontend, TypeScript).

@docs/ROSTER.md
@docs/PROJECT.md
