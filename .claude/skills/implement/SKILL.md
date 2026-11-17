---
name: implement
description: Role for making an approved code change in this project, one change at a time, with tests. Use after a plan is approved or for a small clear fix.
---

1. Confirm the task in `docs/ROSTER.md` (or the user's request). Read the affected files and the nearest directory `CLAUDE.md`.
2. Edit only what is needed; preserve behavior; update imports and interfaces. One unrelated change at a time.
3. Adding a tool: follow "Adding a tool" in root `CLAUDE.md`.
4. Add or update tests, then switch to the `test` role to run them. Never claim success unverified.
5. Do not touch `.env`, secrets or unrelated files. No commit or push without permission.
