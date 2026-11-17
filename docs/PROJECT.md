# Project knowledge

Shared memory across sessions. Read at session start (imported by root `CLAUDE.md`). Update at session end via `docs-sync`. Keep it short: summarize or delete stale entries; do not duplicate what the code or the `CLAUDE.md` files already say.

Tags: **[verified]** checked in code/tests/docs (cite where) · **[assumed]** believed, not checked · **[unknown]** open.

## Current state
- Branch `main`. Last commits: `892bd56` docs, `8f4c300` live input/jobs/voice/web/hybrid RAG. [verified: git log, 2026-11-17]
- Uncommitted at 2026-11-17 start: edits in `memories/store.py`, `memories/table.py`, `retrieval.py`, `tools.py`, `turns.py`, `voice/text.py`, `data/__init__.py`, and many `tests/*`. Purpose not yet recorded. [verified: git status]
- Test status: full suite 520 tests OK, 11 skipped, 2026-11-17, on branch `feature/F019-webui-typescript-sessions` (includes 31 frontend `node --test` tests run through `tests/test_webui_frontend.py`). [verified: unittest discover]
- F019 (web UI rewrite) is on `feature/F019-webui-typescript-sessions`, not merged. Unrelated uncommitted edits (F017 DB port/`memories.check`, `retrieval.py`, `turns.py`, voice, tests) were left unstaged on purpose; doc hunks about them in `README.md`/`CLAUDE.md`/`PROJECT.md` are also still unstaged. [verified: git status]

## Decisions
Format: `date — decision — why — alternatives rejected`.
- 2026-11-17 — Instructions split: root `CLAUDE.md` + `voice/CLAUDE.md` + `memories/CLAUDE.md` + skills; roles are skills — keeps always-loaded context small; solo project needs no agent fleet.
- 2026-11-17 — Removed dead `app.py`, `chains/rag_chain.py`, `data/rag_data.py` (no references outside each other; RAG is the `search_document` tool) — less misleading code — keeping them rejected.
- 2026-11-17 — `TOKENIZER_MAP` + `model_name()` moved to `llm_options.py` so a missing or unsupported `MODEL` fails with a fix-it message and is unit-testable (`dependency.py` cannot be imported in tests) — keeping a bare `os.environ["MODEL"]` KeyError rejected.

- 2026-11-17 — Web chat (F018) as `webui/` on a stdlib HTTP+SSE server, loopback + token, own `ChatSession` over `turns.run_turn` — no new dependencies, no change to `main.py`/`ChatLoop` (its output mixes CLI prompts into text) — FastAPI rejected (two new deps for ~200 lines of routing); reusing `ChatLoop` rejected.
- 2026-11-17 — F019 split the web chat: Python API `backend/` (`python -m backend`, same stdlib server + security) and TypeScript frontend `webui/` (no framework/bundler; `tsc` only, tests via `node --test` on `.ts`), with many sessions (`SessionManager`; one LangGraph thread each; shared semaphore, default one turn at a time; in memory) — user asked for frontend/backend split, TypeScript and multiple sessions; one turn at a time because qwen3:8b runs ~4 tok/s on this GPU and the shared HF tokenizer's thread safety is unverified — React/Vite rejected (many deps for one page); SQLite persistence rejected for now (new dependency); F018's Python `webui/` removed (never committed; backup was only in a session scratchpad).
- 2026-11-17 — New features are entered in `feature-lists.json` with status `open` before planning (user rule; `new-feature` skill step 0).

## Verified findings
- Windows: `HTTPServer.allow_reuse_address` (SO_REUSEADDR) lets a second process bind the same port silently; `backend.server.WebServer` turns it off (`tests/test_backend_server.py::LoopbackTests`). A refused HTTP body must be drained or the 413 is lost to a connection reset (`Handler._discard`).
- Headless Chrome `--virtual-time-budget` / `--dump-dom` never finish on the web chat (an open EventSource keeps the network busy); drive it over the DevTools protocol instead (Node's built-in `WebSocket`). [verified 2026-11-17]
- (add: finding — evidence file:line or test)

## Dependencies and conflicts
- `dependency.py` must not import `graph` (circular). [verified: CLAUDE.md, not re-checked]
- New tool touches `tools.py` (`TOOLS`, `NEEDS_APPROVAL`) and `chains/code_chain.py` prompt; a test checks the prompt's tool names exist.
- Port `localhost:5432` is shared with another process (memory connect failures). [reported in CLAUDE.md]

## Open questions
- What are the uncommitted edits (from before 2026-11-17) for? They pass the tests, but their purpose is not recorded.
- `tools.read_file`: `end_line=0` reads the whole file (`end_line or len(lines)`). Harmless; left as is.
- Are Superpowers, Serena and Pyright installed here? [unknown]
