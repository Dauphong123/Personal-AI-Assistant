---
name: docs-sync
description: Role for keeping docs in step with code and saving context between sessions. Use after code changes pass tests and at the end of every working session.
---

1. `git diff` / `git status`: what actually changed.
2. Update only affected docs: the nearest directory `CLAUDE.md`, root `CLAUDE.md` (only if always-needed), `README.md`, `.env.example`. Verify each claim against the code; fix stale statements.
3. `docs/PROJECT.md`: refresh Current state, add Decisions and verified Findings (with evidence), update Dependencies/conflicts, move answered questions out of Open questions, add new unknowns. Prune stale entries.
4. `docs/ROSTER.md`: update task statuses; remove done tasks.
5. Keep entries short and non-duplicated. Report what was updated. Do not commit without permission.
