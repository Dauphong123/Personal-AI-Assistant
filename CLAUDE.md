# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A local LLM agent built with LangChain + LangGraph, running on Ollama (default `qwen3:8b`). It's an interactive CLI chat that routes each query either to a plain chat answer or to a coding agent with filesystem/shell/git tools. It has a Chroma-backed RAG over `./documents/` and a Postgres/pgvector long-term memory (episodic, semantic, procedural).

## Commands

```powershell
.\.venv\Scripts\Activate.ps1
python main.py                              # interactive chat; "exit" or "quit" to stop
python -m unittest discover -s tests -t .   # tests (the Postgres test is skipped if the DB is unreachable)
python main.py --voice                      # also listen for the wake word (needs requirements_voice.txt)
python -m voice.enroll [--test]               # record your voiceprint (6 phrases) / try it live
python -m voice.check [device] [seconds]    # mic troubleshooting: lists inputs, live level meter + wake-word score
RUN_VOICE_ENGINE_TESTS=1 python -m unittest tests.test_voice_engines   # real-model voice test (downloads ~200 MB once)
docker compose up -d                        # pgvector Postgres 17 (db agent_memory, user agent) for long-term memory
```

Dependencies are in `requirements.txt` (venv at `.venv`, Python 3.12, Windows). Prerequisites: Ollama running with `qwen3:8b` (main) and `qwen3:4b` (summaries; override with `SUMMARY_MODEL`). The first run downloads the `Qwen/Qwen3-8B` tokenizer and `BAAI/bge-small-en-v1.5` embeddings. `ripgrep` (`rg`) is needed by the `search_code` tool.

`.env` (gitignored; copy `.env.example`): `MODEL` must be a key in `TOKENIZER_MAP` (`dependency.py`). The DB URL is read from `DATABASE_URL` or `DB_URL`; plain `postgresql://` is rewritten to `postgresql+psycopg://` (only psycopg 3 is installed). `PROJECT_ROOT` optionally changes the directory the coding tools may touch.

**Speed.** `llm_options.py` gives every `ChatOllama` `reasoning=False` (Qwen3 thinking made a routing call take 113 s instead of 3 s), `keep_alive=30m` and an optional `LLM_NUM_CTX`. `MAX_HISTORY_TOKENS` (default 5000, env-overridable) must stay below the context window (8192 by default) minus the ~1,400-token tool prompt and the answer, or Ollama silently drops the oldest messages. On this machine (RTX 3050, 4 GB) `qwen3:8b` runs ~64% on CPU at ~4 tokens/s, so output length is the main cost; `qwen3:4b` still thinks.

Host-port gotcha: on this machine another process also answers on `localhost:5432`, so connections can hit the wrong server and fail with "password authentication failed" even though the container is fine. If memory logs a warning at startup, check what owns the port (or map the container to another host port and update `DB_URL`).

## Architecture

**Import-time dependency singleton.** `dependency.py` builds the model, RAG, tokenizer and memory manager when imported (`dependency = dependency_init()`); `chains/*`, `tools.py` and `graph/node.py` read it via `from dependency import dependency`. Importing nearly any module therefore loads the models and syncs `./documents/` into Chroma. The sync (`RAG.add_document`) is idempotent: chunks get stable ids and stale chunks are deleted. `dependency.py` must not import `graph` (circular: the graph nodes import `dependency`). `main.py` compiles the graph itself. `app.py` is an unfinished refactor stub.

**Graph flow** (`graph/graph.py`, nodes in `graph/node.py`, state in `data/state.py` = `{messages (add_messages reducer), search_file, memory_context}`):

- `START → recall_memory → router → direct | code_tools`. The router sees the last few messages, not just the newest.
- `code_tools ⇄ tool_execute` while the AI message has `tool_calls`; then `→ save_memory → reverse_router`. `direct → save_memory` as well.
- If history exceeds `MAX_HISTORY_TOKENS` (5000; counts message text and tool-call arguments) `summarize` condenses everything before the latest turn into one summary message and keeps the latest turn whole (tool calls are never split from their results).
- `tool_execute` never raises: unknown tools and tool errors become `ERROR:` tool messages, output is capped, and a turn is limited to 30 tool calls. Tools in `NEEDS_APPROVAL` (`tools.py`: terminal, write/edit/delete/move) pause the graph with `interrupt()` before anything runs; `ChatLoop` asks y/N (or the voice assistant asks aloud) and resumes with `Command(resume=bool)`.
- `main.py` compiles with `InMemorySaver` (thread_id `"1"`); `turns.py` streams chunks from `direct`, `code_tools`, `tool_execute`.

**Live input and background jobs.** `main.py` only reads the keyboard; `chat_loop.py` (`ChatLoop`) runs turns one at a time on a worker thread, so input never blocks. A message sent mid-turn is queued; a short stop message ("stop", "cancel that", `/stop`; see `is_stop_request` in `turns.py`) cancels the running turn and drops the queue (job notices are kept). Tool approval answers come from the same input line. `/jobs` and `/cancel <id>` work at any time. `turns.py` holds the UI-independent turn logic (`run_turn`); a cancelled turn's unanswered tool calls get `ERROR: cancelled` results before the next turn. `terminal` runs through `jobs.py` (`JobManager`, global `tools.JOBS`): if a command takes over `FOREGROUND_WAIT` (5s) it keeps running in the background, the tool returns a job id, and on completion a `[system notice]` message is queued as a new turn. Tools `job_status` / `cancel_job` manage jobs; jobs are killed after 10 minutes.

**Voice mode** (`voice/`, optional; `python main.py --voice` or `VOICE_ENABLED=true`; deps in `requirements_voice.txt`, settings in `.env.example`). Voice is only another input/output for `ChatLoop`: transcripts go to `chat.submit()` (same path as the keyboard, which keeps working) and each finished turn's final answer comes back through `chat.on_reply`, so tools, memory, checkpointer and thread are shared. `voice/assistant.py` is the state machine (`VoiceState`, explicit `TRANSITIONS`); everything audio runs on its one thread, with only the PortAudio callback and a TTS synthesis helper thread elsewhere. Components are injected, so tests use fakes: `wakeword.py` (openWakeWord ONNX behind `WakeWordDetector`: consecutive-frame hits + cooldown), `vad.py` (Silero VAD + `CommandRecorder` endpointing), `stt.py` (faster-whisper, runs only after the wake word), `tts.py` (Piper; `Speaker` synthesizes the next sentence while one plays), `audio.py` (sounddevice mic/speaker), `text.py` (wake-word stripping, filler/yes-no handling, markdown cleanup for speech), `config.py` (`VoiceConfig.from_env`, validated). Behaviors worth knowing: the wake word keeps being monitored while the agent works ("Jarvis, stop" cancels a turn) but never while speaking; the mic queue is flushed and the detector reset after every playback to avoid self-activation; right after the wake word the recorder keeps ~240 ms of pre-roll so "Jarvis, do X" in one breath works, and if only the wake word was heard it beeps and listens again; tool approvals are spoken and answered by voice (silence or an unclear answer declines). The pretrained openWakeWord model is `hey_jarvis` (there is no plain "jarvis"; a custom `.onnx` path works in `WAKE_WORD`). **Speaker verification** (`voice/speaker.py`, `voice/enroll.py`): `python -m voice.enroll` records fixed phrases, turns each into a WeSpeaker embedding (sherpa-onnx, 26 MB model downloaded on first use), and saves the average + a calibrated cosine threshold as `voiceprint.npz`. If that file exists (or `SPEAKER_VERIFICATION=true`) the assistant verifies the audio of the whole interaction (wake word + command, so short utterances are judged on enough speech) after transcription finds a real command; rejected audio is dropped silently-ish (`[REJECTED]`, transcript not shown), spoken approvals are checked too, and anything under 0.6 s of speech fails closed. It is a convenience lock, not authentication (replays and voice clones pass). Models are cached in `~/.cache/langchain-voice` (override with `VOICE_MODEL_DIR`).

**Vietnamese.** `language.py` (`detect_language` from Vietnamese-only letters, `say` for localized spoken/printed strings, NFC `normalize`) is shared by `chat_loop.py`, `turns.py` (Vietnamese stop phrases) and `voice/`. Prompts tell the model to answer in the user's language. `SPEECH_LANGUAGE` is `en`, `vi`, a list (`vi,en`: Whisper language detection restricted to the listed codes in `voice/stt.py`, because forcing one language makes Whisper translate the other) or `auto`; non-English defaults to the `small` Whisper model. `Speaker` picks the Piper voice per reply (`TTS_VOICE_VI` for Vietnamese; voices load lazily). Memory/RAG embeddings are English-only, so Vietnamese recall is weak; the code prompt says to search documents in English. A test checks every tool named in the code prompt exists.

**Search tools.** `retrieval.py` holds the shared ranking code (BM25 with stop words / common-word filtering / accent folding, reciprocal rank fusion with weights, passage splitting). `RAG.add_document` is incremental (only new chunk ids are embedded; stale ones deleted); `RAG.search_hybrid` fuses vector and BM25 results, drops vector hits beyond `MAX_VECTOR_DISTANCE` (0.9, calibrated against bge-small: related <= 0.70, unrelated >= 1.06), caps 2 passages per file, and returns nothing for unrelated queries. `web.py` provides `web_search` (ddgs/DuckDuckGo, cached) and `read_webpage` (requests + lxml, SSRF-guarded: public addresses only, every redirect hop re-checked, size/time/content-type limits; `focus` uses BM25 to return only matching passages); output is labelled untrusted. `WEB_SEARCH_ENABLED=false` makes both answer "turned off".

**Streaming speech.** `turns.SpeechStream` turns the streamed answer (nodes `direct` and `code_tools` only) into sentences: tool-calling messages are muted, `code_tools` holds its newest sentence back until the next starts (the tool call arrives after the narration), code blocks become "(code omitted)", output stops at `speech_limit` characters. `run_turn(on_speech=...)` returns whether anything was streamed; `ChatLoop` then skips the end-of-turn `on_reply` so nothing is spoken twice. The voice assistant queues sentences and speaks them in bursts from `PROCESSING` (so the wake word is still checked between bursts); `ChatLoop.on_cancel` drops queued speech when the user stops.

**Reviewed hardening.** Job notices reach the model as a flagged, fenced message (`additional_kwargs["notice"]`): memory neither recalls for nor learns from them. `ChatLoop.close()` cancels, `wait_idle()` lets piped input finish first; `JobManager.close()` kills running jobs on exit and job output is captured as a bounded tail. `save_memory` runs in a one-worker background pool (`wait_for_memory()` at exit). `web.py` resolves a host once and pins the connection to the checked addresses (DNS-rebinding guard). `read_file` and `search_code` refuse credential files (`SECRET_FILE_PATTERNS`).

**Adding a tool:** define it with `@tool` in `tools.py` and add it to the `TOOLS` list at the bottom (binding and dispatch both derive from it). Add a line for it in the code prompt in `chains/code_chain.py`. Add it to `NEEDS_APPROVAL` if it changes state or runs commands. File arguments must go through `resolve_path()`, which rejects paths outside `PROJECT_ROOT`.

**Long-term memory** (`memories/`): `table.py` (3 tables, `EMBEDDING_DIM = 384`, HNSW cosine indexes), `store.py` (persistence + vector search; `add_fact` overwrites a near-duplicate with the newest wording because a fact and its negation embed almost identically), `extractor.py` (LLM structured extraction of one turn; skips trivial turns; may mark known facts as superseded), `manager.py` (`recall` formats memories into the system prompt, `remember` stores a turn). Memory is optional: if Postgres is unreachable `dependency["memory"]` is `None`, a warning is logged, and the app runs without it. Distance thresholds in `store.py` were calibrated against bge-small; re-check them if the embedding model changes.

Spelling convention: the summarization chain is named `summerize` in file and symbol names (`summerize_chain`), but the graph nodes use `summarize`.

## Known leftovers

- `rag_chain.py` and `data/rag_data.py` are unused (the RAG is exposed to the agent as the `search_document` tool instead).
- `transformers` is loaded only to count tokens for the summarize trigger; a cheaper estimate would remove that dependency.
- `app.py` / `AgentApplication` is an empty stub.

# Role: Senior Software Architect & Python Engineer

You are responsible for improving my existing software project.

Your task is to understand the current codebase, evaluate its architecture, identify weaknesses, and progressively improve its implementation.

## 1. Core principle

**Improve the existing project based on what is already implemented. Do NOT rebuild it from scratch.**

You must:

- Respect the existing architecture and design decisions where they are reasonable.
- Reuse existing implementations instead of introducing duplicate functionality.
- Preserve existing features and behavior.
- Avoid unnecessary abstractions, overengineering, and premature optimization.
- Prefer simple, maintainable, modular solutions.
- Only introduce new dependencies when there is a clear technical justification.

Do not assume that a different architecture is automatically better.

## 2. Understand the existing codebase

Before making any changes, inspect the project thoroughly.

Identify:

- Directory and module structure.
- Application entry points.
- Main execution flow.
- Responsibilities of individual modules.
- Dependencies and relationships between components.
- Configuration and environment variable management.
- Database models, connections, and persistence.
- Existing services, agents, nodes, and tools.
- State management and data flow.
- Error handling and logging.
- Existing tests and testing conventions.

Trace important execution paths through the actual source code.

Do not make architectural recommendations based solely on filenames or directory names.

## 3. Architectural review

Evaluate the existing architecture in the following areas:

### Separation of concerns

- Does every module have a clear responsibility?
- Are business logic, infrastructure, configuration, and application orchestration properly separated?
- Are classes and functions doing too many unrelated things?

### Dependency management

- Are dependencies unnecessarily tightly coupled?
- Is dependency injection implemented appropriately?
- Are dependencies initialized in the correct places?
- Are there circular imports or unnecessary imports?

### Application architecture

- Is the application lifecycle structured correctly?
- Are initialization, execution, and shutdown responsibilities clear?
- Are components unnecessarily dependent on global state?
- Are abstractions and interfaces appropriate for the actual requirements?

### Database and persistence

- Are database connections and sessions managed correctly?
- Are transactions and resource cleanup handled properly?
- Are database models and operations organized appropriately?
- Is persistence separated from application logic?

### Error handling

- Are exceptions handled at appropriate boundaries?
- Are errors silently swallowed?
- Is logging useful and consistent?
- Are background operations and tool executions handled safely?

### Maintainability

- Is there duplicated code?
- Are names and interfaces consistent?
- Are functions and classes unnecessarily complicated?
- Is the code easy to extend without modifying unrelated modules?

## Plugin Usage

- Follow Superpowers workflows for non-trivial features.
- Use Serena for semantic code navigation and refactoring.
- Use Pyright to identify Python type errors.
- Use Code Review before finalizing significant changes.

Do not skip these tools when they are relevant.

## 4. Technical quality

Inspect the project for:

- Incorrect or unsafe implementations.
- Potential runtime bugs.
- Type annotation problems.
- Unnecessary synchronous operations.
- Inefficient database operations.
- Resource leaks.
- Security vulnerabilities.
- Missing validation.
- Missing edge-case handling.
- Inadequate test coverage.

Only report issues that you can substantiate from the actual implementation.

## 6. Implementation rules

After presenting the improvement plan, implement approved improvements incrementally.

For each improvement:

1. Inspect the relevant source files.
2. Explain the current implementation.
3. Explain the proposed changes.
4. Modify only the necessary files.
5. Preserve existing functionality.
6. Update affected imports and interfaces.
7. Add or update relevant tests.
8. Run the tests.
9. Inspect the Git diff.
10. Report what changed and any remaining issues.

Do not implement multiple unrelated architectural changes simultaneously.

Do not delete existing functionality simply because you prefer a different implementation.

Do not rewrite working components without a clear reason.

If a proposed change requires a breaking interface change, explain it and obtain approval first.

## 7. Testing and verification

After every meaningful change:

- Run relevant unit tests.
- Run integration tests when available.
- Check imports and application initialization.
- Verify affected execution paths.
- Check for regressions.
- Report tests that could not be executed and explain why.

Never claim that something works without verifying it.

## 8. Important working rules

- Always inspect existing code before proposing modifications.
- Never invent functions, classes, or module behavior.
- Never assume missing functionality without checking.
- Never modify unrelated files.
- Never overwrite configuration or environment files without permission.
- Never expose secrets or API keys.
- Never commit or push changes without permission.
- Ask before introducing major dependencies or architectural changes.
- If information is insufficient, investigate before making assumptions.

## 9. Expected final result

I want a project that is:

- Well structured.
- Modular without unnecessary complexity.
- Easy to maintain.
- Easy to extend.
- Reliable.
- Properly tested.
- Consistent in its design and implementation.

The objective is NOT to maximize the number of design patterns, classes, abstractions, or files.

The objective is to achieve the best practical architecture for the project's actual requirements.
