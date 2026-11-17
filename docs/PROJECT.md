# Project knowledge

Shared memory across sessions. Read at session start (imported by root `CLAUDE.md`). Update at session end via `docs-sync`. Keep it short: summarize or delete stale entries; do not duplicate what the code or the `CLAUDE.md` files already say.

Tags: **[verified]** checked in code/tests/docs (cite where) · **[assumed]** believed, not checked · **[unknown]** open.

## Current state
- Branch `main`. Last commits: `892bd56` docs, `8f4c300` live input/jobs/voice/web/hybrid RAG. [verified: git log, 2026-11-17]
- Uncommitted at 2026-11-17 start: edits in `memories/store.py`, `memories/table.py`, `retrieval.py`, `tools.py`, `turns.py`, `voice/text.py`, `data/__init__.py`, and many `tests/*`. Purpose not yet recorded. [verified: git status]
- Test status: full suite 576 tests OK, 11 skipped, 2026-11-17, on branch `feature/F020-webui-handshake` (includes 45 frontend `node --test` tests run through `tests/test_webui_frontend.py`). [verified: unittest discover]
- F019 (web UI rewrite) is on `feature/F019-webui-typescript-sessions`; F020 (handshake, web job notices, web voice) on `feature/F020-webui-handshake`, branched from F019. Neither is merged; merge F019 first. Unrelated uncommitted edits (F017 DB port/`memories.check`, `retrieval.py`, `turns.py`, voice, tests) were left unstaged on purpose; doc hunks about them in `README.md`/`CLAUDE.md`/`PROJECT.md` are also still unstaged. [verified: git status]
- F021 (memory retrieval) on `feature/F021-memory-retrieval`, branched from F020 plus `26c45e5` (memory tests, DB check on port 15432), because `main` lacks them; merge order F019 -> F020 -> F021. [verified: git log]

## Decisions
Format: `date — decision — why — alternatives rejected`.
- 2026-11-17 — Instructions split: root `CLAUDE.md` + `voice/CLAUDE.md` + `memories/CLAUDE.md` + skills; roles are skills — keeps always-loaded context small; solo project needs no agent fleet.
- 2026-11-17 — Removed dead `app.py`, `chains/rag_chain.py`, `data/rag_data.py` (no references outside each other; RAG is the `search_document` tool) — less misleading code — keeping them rejected.
- 2026-11-17 — `TOKENIZER_MAP` + `model_name()` moved to `llm_options.py` so a missing or unsupported `MODEL` fails with a fix-it message and is unit-testable (`dependency.py` cannot be imported in tests) — keeping a bare `os.environ["MODEL"]` KeyError rejected.

- 2026-11-17 — Web chat (F018) as `webui/` on a stdlib HTTP+SSE server, loopback + token, own `ChatSession` over `turns.run_turn` — no new dependencies, no change to `main.py`/`ChatLoop` (its output mixes CLI prompts into text) — FastAPI rejected (two new deps for ~200 lines of routing); reusing `ChatLoop` rejected.
- 2026-11-17 — F019 split the web chat: Python API `backend/` (`python -m backend`, same stdlib server + security) and TypeScript frontend `webui/` (no framework/bundler; `tsc` only, tests via `node --test` on `.ts`), with many sessions (`SessionManager`; one LangGraph thread each; shared semaphore, default one turn at a time; in memory) — user asked for frontend/backend split, TypeScript and multiple sessions; one turn at a time because qwen3:8b runs ~4 tok/s on this GPU and the shared HF tokenizer's thread safety is unverified — React/Vite rejected (many deps for one page); SQLite persistence rejected for now (new dependency); F018's Python `webui/` removed (never committed; backup was only in a session scratchpad).
- 2026-11-17 — New features are entered in `feature-lists.json` with status `open` before planning (user rule; `new-feature` skill step 0).
- 2026-11-17 — F020: frontend/backend handshake (`POST /api/handshake`, `PROTOCOL` constant on both sides) gates features; job notices routed by a `jobs.JOB_OWNER` contextvar stamped on each `Job` (langgraph copies contexts into node threads, `langgraph/pregel/_executor.py:64`); web voice is push-to-talk through backend STT/TTS (`--voice`) — passing the session through graph config/tool args rejected (touches graph and tool signatures); browser Web Speech API rejected (sends audio to a cloud service, no Vietnamese Piper voice); WebSocket audio streaming rejected (no new deps, a recording per message is enough).
- 2026-11-17 — F021: semantic recall = always-on profile facts (categories `user_profile`/`user_preference`/`environment`, cap 12) + hybrid fact search (vector + BM25, `retrieval.py`) + follow-up context (previous question + last answer); extractor at temperature 0 with fixed categories, a code secret filter, and a review of stored facts near each new fact — measured by `tests/test_memory_eval.py` (15/30 -> 33/35) — LLM query rewriting rejected (~4 tok/s), Postgres full-text rejected (English stemming; BM25 already folds accents), bge-base rejected (768-dim schema change). Multilingual memory embeddings (`intfloat/multilingual-e5-small`) proposed, not approved yet.

## Verified findings
- Windows: `HTTPServer.allow_reuse_address` (SO_REUSEADDR) lets a second process bind the same port silently; `backend.server.WebServer` turns it off (`tests/test_backend_server.py::LoopbackTests`). A refused HTTP body must be drained or the 413 is lost to a connection reset (`Handler._discard`).
- Windows: closing a socket with an unread request body resets the connection, so the client gets `ConnectionAbortedError` instead of the 401/403; `backend/server.py` drains refused POST bodies (`tests/test_backend_server.py::SecurityTests::test_refused_post_is_answered_not_reset`, was 1-in-150 flaky). [verified 2026-11-17]
- `faster_whisper.audio.decode_audio` (faster-whisper 1.2.1) fails with PyAV 19.0.2 (`open() got an unexpected keyword argument 'metadata_errors'`); `backend.voice.decode_recording` uses PyAV directly (`tests/test_backend_voice.py::RealDecodeTests`). The keyboard voice mode is unaffected (passes raw samples). [verified 2026-11-17]
- Headless Chrome `--virtual-time-budget` / `--dump-dom` never finish on the web chat (an open EventSource keeps the network busy); drive it over the DevTools protocol instead (Node's built-in `WebSocket`). [verified 2026-11-17]
- Long-term memory saved episodes but no facts (DB had 4 episodes, 0 facts): the extractor prompt listed only preferences/project/environment, so qwen3:8b skipped "my name is Dat", and `is_trivial_turn` (< 3 words) dropped "I'm Dat" before the model. Fixed in F021 step 0 (`tests/test_memory.py::TrivialTurnTests`, `ExtractionPromptTests`). [verified 2026-11-17]
- qwen3:8b extraction at the chat default temperature (~0.6) varied between runs and once stored "The user's OpenAI key is sk-..." despite the prompt; temperature 0 plus the `looks_secret` filter fixed both (scratchpad eval 13/13). bge-small blurs short facts: "The user uses Vim" is closer to "The user is a student" (0.23) than to "The user's preferred code editor is VS Code" (0.36); with the same subject wording ("The user's code editor is ...") they are 0.09 apart. Vietnamese questions score 0.39-0.59 against English facts. [verified 2026-11-17, measured with bge-small-en-v1.5]
- Memory retrieval eval (`tests/test_memory_eval.py`): 15/30 -> profile 26/30 -> (35 cases) hybrid 32/35 -> follow-up 33/35. Misses: "con mèo của tôi tên gì?" (Vietnamese) and "suggest a dinner I'd like" (needs inference). [verified 2026-11-17]
- (add: finding — evidence file:line or test)

## Dependencies and conflicts
- `dependency.py` must not import `graph` (circular). [verified: CLAUDE.md, not re-checked]
- New tool touches `tools.py` (`TOOLS`, `NEEDS_APPROVAL`) and `chains/code_chain.py` prompt; a test checks the prompt's tool names exist.
- Port `localhost:5432` is shared with another process (memory connect failures). [reported in CLAUDE.md]

## Open questions
- Vietnamese memory recall stays weak (eval miss `vi-cat`): switch memory embeddings to `intfloat/multilingual-e5-small` (384-dim [assumed], re-embed rows, recalibrate thresholds)? Awaiting user decision.
- What are the uncommitted edits (from before 2026-11-17) for? They pass the tests, but their purpose is not recorded.
- `tools.read_file`: `end_line=0` reads the whole file (`end_line or len(lines)`). Harmless; left as is.
- Are Superpowers, Serena and Pyright installed here? [unknown]
