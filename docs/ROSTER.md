# Roster

Solo project. One Claude session wears one role at a time; a role is a skill in `.claude/skills/`, loaded only when needed. No extra agents. Spawn a subagent only for broad read-only search or a fresh-eyes review.

| Role (skill) | Owns | Use when | Hands off to |
|---|---|---|---|
| `architect` (+ `architecture-review`) | Design decisions, module boundaries, `docs/PROJECT.md` Decisions | New feature shape, refactor, "is this the right structure?" | `implement` |
| `implement` | Source edits, one change at a time | A plan is approved | `test` |
| `test` | `tests/`, running suites, coverage gaps | After any code change; before claiming done | `debug` on failure, `docs-sync` on pass |
| `debug` | Root-cause investigation | A test, run or behavior is wrong | `implement` with a verified cause |
| `docs-sync` | `CLAUDE.md` files, `README.md`, `docs/` | After code changes; end of every session | (done) |

## Active tasks
Format: `- [status] task — role — depends on — files`. Status: todo, doing, blocked, done (remove done after logging the outcome in PROJECT.md).

- (none yet)

## Rules
- Domain-specific notes live in the nearest directory `CLAUDE.md` (`voice/`, `memories/`); this roster only points to them.
- Cross-area change (e.g. `tools.py` + `chains/code_chain.py` + `tests/`): list every touched area in the task line first.
- Unfamiliar area: read its `CLAUDE.md` and the code before editing.
