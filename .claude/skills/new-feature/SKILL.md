---
name: new-feature
description: Workflow for adding ANY new feature to this project - one feature per Claude session on its own feature/F0xx branch; recommend steps first, then per step write tests, implement, repeat until green, commit and push the branch; record the feature in feature-lists.json. Use whenever the user asks for a new feature or capability.
---

Follow in order. Do not skip phases. Root `CLAUDE.md` rules still apply (no unrelated edits, no `.env` changes), with one exception: **this workflow is the user's standing permission to commit and push** the feature's own changes (Phase 2 step 8, Phase 3).

## Rule: one feature = one Claude session
- A session works on exactly one feature. Context from an earlier feature must not leak into the next.
- **At start:** if this session has already planned, implemented or committed a different feature, stop. Tell the user: "This session already handled F0xx. Run `/clear` (or open a new Claude session) and ask for the new feature there." Do no work on the new feature.
- **Resuming:** a new session continues an unfinished feature only from what is on disk - `feature-lists.json` (status `open`/`planned`/`in_progress`, step statuses, `branch`), `docs/PROJECT.md`, `git log`. Check out the feature's `branch` first (`git switch <branch>`; if only on the remote, `git fetch` then `git switch <branch>`). Never rely on memory of an earlier conversation.
- **At end:** after Phase 3, do not start another feature, even if asked in the same message. Say: "F0xx is done and pushed. Run `/clear` or open a new session for the next feature."

## Phase 1 - Recommend steps (before any code)
0. **Open the feature first.** Before reading code or planning, add the feature to `feature-lists.json` with the next id, status `open`, `created` set, empty `steps`. (User rule, 2026-11-17.)
1. Read `feature-lists.json` (avoid duplicating or conflicting with existing features; pick up an unfinished one if it is this feature), `docs/PROJECT.md`, the nearest directory `CLAUDE.md`, and the real source the feature touches.
2. Check git: `git status`, `git branch --show-current`, `git remote -v`. Note any pre-existing uncommitted changes - they are **not** part of this feature and must never be staged by it. If a file the feature must edit already has unrelated uncommitted changes, ask the user how to handle it before planning.
3. Present a numbered plan: a short goal, then small ordered steps. Each step lists: what it does, files touched, and the test that proves it (name + what it asserts). Flag breaking interface changes, new dependencies, or architecture changes - ask first for those.
4. Wait for user approval/edits.
5. **Create the feature branch** from an up-to-date `main`: `git switch main`, `git pull --ff-only`, then `git switch -c feature/F0xx-<short-slug>` (lowercase, hyphens, e.g. `feature/F019-pdf-ingest`). Uncommitted unrelated changes travel with the switch; leave them unstaged. If `main` cannot be pulled cleanly or the switch is refused, report and ask - do not stash, reset or discard anything.
6. Update the feature's `open` entry in `feature-lists.json` to status `planned`, `branch` set, and the approved steps (all `todo`).

## Branch rules
- One branch per feature: `feature/F0xx-<slug>`. All of the feature's commits go there; never commit feature work on `main`.
- Never merge into `main`, rebase, or force-push without the user asking. At the end, offer a pull request instead (Phase 3).

## Phase 2 - Per step loop (TDD, commit + push per step)
For each step, in order:
1. Set the step `in_progress` (and the feature `in_progress`) in `feature-lists.json`.
2. **Write tests first** in `tests/` (unittest, reuse `tests/stubs.py`; no real Ollama/DB/network). Run them and confirm they fail for the right reason.
3. **Implement** the minimum code to pass (follow "Adding a tool" in `CLAUDE.md` when relevant).
4. **Run** the step's tests: `python -m unittest tests.test_x -v` (activate `.venv` first).
5. If red: fix the code (or the test only if the test itself is wrong, and say so), re-run. Repeat until green. If stuck after 3 failed rounds, switch to the `debug` skill; do not guess.
6. When green, run the full suite `python -m unittest discover -s tests -t .` to catch regressions. Fix any regression before moving on. **Never commit on red.**
7. Mark the step `done` with the test names and the date.
8. **Commit and push** this step:
   - `git diff` the step's files and check no secrets, `.env`, or unrelated edits.
   - Stage only the step's files by explicit path (`git add <files> feature-lists.json`) - never `git add -A` / `git add .`.
   - `git commit -m "feat(F0xx): step N - <title>"` ending with the attribution line from the system reminder.
   - Confirm `git branch --show-current` is the feature's branch, then push: first push `git push -u origin feature/F0xx-<slug>`, later ones `git push`.
   - If the push fails (auth, rejected, no network): do not force-push, do not rebase or reset. Report the exact error, leave the commit local, and continue only if the user says so.

## Phase 3 - Finish
1. Run the full suite once more; inspect `git status` and `git diff` for anything left unstaged from this feature.
2. Set feature status `done` (or `blocked` with a reason), fill `completed`, `files`, `tests`, and `commit` (last commit hash) in `feature-lists.json`.
3. Run the `docs-sync` skill (update `CLAUDE.md`/`docs/PROJECT.md` if behavior changed) so the next session can resume without this conversation.
4. Commit and push the bookkeeping: `git add feature-lists.json <changed docs>`, `git commit -m "docs(F0xx): finish <name>"`, `git push`. Same failure rule as Phase 2 step 8.
5. Offer a pull request `feature/F0xx-<slug>` → `main` (`gh pr create --base main --head feature/F0xx-<slug>`, body ending with the PR attribution line from the system reminder). Create it only if the user says yes; do not merge it.
6. Report: branch, what changed, commits pushed (hashes), PR link if created, test results (and any that could not run, with why), remaining issues. Then end the session per the one-feature rule.

## feature-lists.json
Repo root. Single source of truth for features and the hand-off between sessions. Keep valid JSON; update it at every state change.

```json
{
  "features": [
    {
      "id": "F001",
      "name": "short name",
      "description": "one sentence",
      "status": "open | planned | in_progress | done | blocked",
      "created": "YYYY-MM-DD",
      "completed": null,
      "branch": "feature/F001-short-slug",
      "commit": "short hash of the last pushed commit, else null",
      "files": ["tools.py"],
      "tests": ["tests/test_tools.py::TestX"],
      "steps": [
        {"n": 1, "title": "...", "status": "todo | in_progress | done | blocked", "tests": ["test_name"], "commit": "short hash or null"}
      ],
      "notes": ""
    }
  ]
}
```
IDs are sequential (`F001`, `F002`, ...). Never delete entries; mark `blocked` with a reason instead.
