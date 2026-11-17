# Project knowledge

Shared memory across sessions. Read at session start (imported by root `CLAUDE.md`). Update at session end via `docs-sync`. Keep it short: summarize or delete stale entries; do not duplicate what the code or the `CLAUDE.md` files already say.

Tags: **[verified]** checked in code/tests/docs (cite where) · **[assumed]** believed, not checked · **[unknown]** open.

## Current state
- Branch `main`. Last commits: `892bd56` docs, `8f4c300` live input/jobs/voice/web/hybrid RAG. [verified: git log, 2026-11-17]
- Uncommitted at 2026-11-17 start: edits in `memories/store.py`, `memories/table.py`, `retrieval.py`, `tools.py`, `turns.py`, `voice/text.py`, `data/__init__.py`, and many `tests/*`. Purpose not yet recorded. [verified: git status]
- Test status: full suite 471 tests OK, 11 skipped, 2026-11-17 (after removing leftovers and adding the `MODEL` check). The earlier uncommitted edits were green too (468 OK). [verified: unittest discover]

## Decisions
Format: `date — decision — why — alternatives rejected`.
- 2026-11-17 — Instructions split: root `CLAUDE.md` + `voice/CLAUDE.md` + `memories/CLAUDE.md` + skills; roles are skills — keeps always-loaded context small; solo project needs no agent fleet.
- 2026-11-17 — Removed dead `app.py`, `chains/rag_chain.py`, `data/rag_data.py` (no references outside each other; RAG is the `search_document` tool) — less misleading code — keeping them rejected.
- 2026-11-17 — `TOKENIZER_MAP` + `model_name()` moved to `llm_options.py` so a missing or unsupported `MODEL` fails with a fix-it message and is unit-testable (`dependency.py` cannot be imported in tests) — keeping a bare `os.environ["MODEL"]` KeyError rejected.

## Verified findings
- (add: finding — evidence file:line or test)

## Dependencies and conflicts
- `dependency.py` must not import `graph` (circular). [verified: CLAUDE.md, not re-checked]
- New tool touches `tools.py` (`TOOLS`, `NEEDS_APPROVAL`) and `chains/code_chain.py` prompt; a test checks the prompt's tool names exist.
- Port `localhost:5432` is shared with another process (memory connect failures). [reported in CLAUDE.md]

## Open questions
- What are the uncommitted edits (from before 2026-11-17) for? They pass the tests, but their purpose is not recorded.
- `tools.read_file`: `end_line=0` reads the whole file (`end_line or len(lines)`). Harmless; left as is.
- Are Superpowers, Serena and Pyright installed here? [unknown]
