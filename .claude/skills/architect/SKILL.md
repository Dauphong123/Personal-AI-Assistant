---
name: architect
description: Role for design work in this project - shaping a feature, refactor or boundary change before code is written. Use when asked "how should we build/structure X" or before a non-trivial change.
---

1. Read `docs/PROJECT.md` (decisions, open questions) and the nearest directory `CLAUDE.md`.
2. Read the real code on the affected path. Reuse what exists; do not rebuild.
3. For a weakness audit, use the `architecture-review` skill.
4. Output a short plan: goal, files touched, approach, alternatives rejected, risks, tests needed. Mark each claim verified / assumed / unknown; check assumptions in code before presenting.
5. Ask the user before breaking interface changes, major dependencies or architecture shifts. Wait for approval.
6. Record the approved decision and rationale in `docs/PROJECT.md` and add tasks to `docs/ROSTER.md`. Next role: `implement`.
