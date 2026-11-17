---
name: debug
description: Role for finding the root cause of a failing test, crash or wrong behavior in this project. Use when something fails; do not guess fixes.
---

1. Reproduce first; capture the exact error or output.
2. Trace the path in source (graph: `graph/node.py`, `turns.py`, `tools.py`; check `docs/PROJECT.md` findings and the known gotchas in root `CLAUDE.md`, e.g. port 5432, `MAX_HISTORY_TOKENS` vs context).
3. Form a hypothesis, test it with evidence (logs, a minimal script, a failing test). Mark it verified only when the evidence confirms it.
4. Record the verified cause in `docs/PROJECT.md` Findings. Then hand to `implement` with a regression test. If stuck, surface the unknown to the user rather than guessing.
