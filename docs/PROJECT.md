# Project knowledge

Shared memory across sessions. Read at session start (imported by root `CLAUDE.md`). Update at session end via `docs-sync`. Keep it short: summarize or delete stale entries; do not duplicate what the code or the `CLAUDE.md` files already say.

Tags: **[verified]** checked in code/tests/docs (cite where) · **[assumed]** believed, not checked · **[unknown]** open.

## Current state
- Branch `main`. Last commits: `892bd56` docs, `8f4c300` live input/jobs/voice/web/hybrid RAG. [verified: git log, 2026-11-17]
- Uncommitted at 2026-11-17 start: edits in `memories/store.py`, `memories/table.py`, `retrieval.py`, `tools.py`, `turns.py`, `voice/text.py`, `data/__init__.py`, and many `tests/*`. Purpose not yet recorded. [verified: git status]
- Test status: not run this session. [unknown]

## Decisions
Format: `date — decision — why — alternatives rejected`.
- 2026-11-17 — Instructions split: root `CLAUDE.md` + `voice/CLAUDE.md` + `memories/CLAUDE.md` + skills; roles are skills — keeps always-loaded context small; solo project needs no agent fleet.

## Verified findings
- (add: finding — evidence file:line or test)

## Dependencies and conflicts
- `dependency.py` must not import `graph` (circular). [verified: CLAUDE.md, not re-checked]
- New tool touches `tools.py` (`TOOLS`, `NEEDS_APPROVAL`) and `chains/code_chain.py` prompt; a test checks the prompt's tool names exist.
- Port `localhost:5432` is shared with another process (memory connect failures). [reported in CLAUDE.md]

## Open questions
- What are the uncommitted edits for, and are tests green with them?
- Known leftovers (unused `rag_chain.py`, `data/rag_data.py`, stub `app.py`): remove or keep?
- Are Superpowers, Serena and Pyright installed here? [unknown]
