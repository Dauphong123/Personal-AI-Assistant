# F036 Claude Code bridge and model switching: design

Agreed in a question-by-question session on 2026-11-20. Feature entry: `feature-lists.json` F036.

## Intent

The user wants the agent to behave like Claude Code on any model they connect, and to use the Claude
plan they already pay for instead of API keys. A Claude plan login may only be used by Anthropic's own
apps, so the app does not call Claude directly: it runs the user's installed Claude Code headless
(`claude -p`) as the "model" for a turn. Local Ollama models stay as they are.

## Goals

- Pick the chat model per session: any installed Ollama chat model (live list) or Claude Code
  (`opus`, `sonnet`, `haiku`, `fable`) with an effort level (`low` … `max`, default `medium`).
- CLI `/model` and a web picker. The choice lives in `model_settings.json` (gitignored); the app
  never writes `.env`. CLI `/model` saves the default; a web pick applies to its session only.
- A Claude Code turn behaves like Claude Code: its own context window, compaction, thinking and tools.
  It keeps its own history per session (`--session-id` / `--resume`).
- Safety unchanged: Claude Code's read, search and web tools run freely; file edits and shell
  (`Edit`, `Write`, `MultiEdit`, `NotebookEdit`, `Bash`) ask the user's y/N, as our own tools do.
- Long-term memory: recall is added before the user's message; memory still learns from the turn.
- Switching from Ollama to Claude Code hands over a recap of the talk Claude has not seen.
- One plain privacy line when switching to Claude Code. Errors (CLI missing, not logged in, plan
  limit) are shown as the answer; the model is kept, no silent fallback.

## Non-goals

- API keys, `langchain-anthropic`, `langchain-google-genai`, token budgets, per-model options page.
- Antigravity (F038, once its CLI is installed; only the desktop app is on this machine).
- Changing the local path: Ollama turns keep the router, tools, approvals and limits.

## Anti-goals

- The app running Claude Code with the user's global hooks, plugins or MCP servers (a hook such as
  the user's "Fact-Forcing Gate" would block headless turns): only project/local settings load.
- Edits or shell commands from Claude Code without the user's approval.
- Writing `.env`.

## Constraints

- Windows, Python 3.12, stdlib only for the bridge (no new package).
- Claude Code 2.1.329 flags (verified with `claude --help`): `-p`, `--output-format stream-json`,
  `--input-format stream-json`, `--include-partial-messages`, `--model`, `--effort`,
  `--session-id <uuid>`, `--resume <id>`, `--setting-sources`, `--strict-mcp-config`,
  `--permission-prompts host`, `--append-system-prompt`, `--verbose`.
- The control-message shape for host permission prompts is [assumed] from the Agent SDK protocol and
  is verified by the user's real run at the end (the user asked that nothing run or be created outside
  the repository, so there is no live probe before building); if it does not work, fall back to
  "read-only plus a per-session allow-edits toggle" and tell the user.
- `MODEL` stays the local base model: router, summaries and memory extraction keep using it.

## Approaches considered

1. **API keys per provider** (LangChain chat classes): our tools and approvals, pay per token,
   plans unused. Dropped: the user wants to use their plans.
2. **Bridge to Claude Code** (chosen): plan login, Claude Code's own behavior; its tools replace ours
   for that turn, so approvals go through its permission channel.
3. **Both**: most work; rejected for now.

## Decided design

- `model_choice.py`: ids `"<ollama name>"` or `"claude-code:<alias>"`; effort for Claude ids only;
  live Ollama list from `/api/tags` (embedding models hidden, newest first, falls back to the last
  saved list plus `MODEL`); `model_settings.json` read and write.
- `claude_bridge.py`: builds the `claude` command, parses stream-json, answers permission requests
  through an `approve` callback, kills the process on cancel, turns failures into clear errors.
- Graph: the `route` node returns `"claude_code"` when the turn's model is Claude Code (no router
  call); new node `claude_code` → `save_memory`. Ollama picks other than `MODEL` get their own cached
  `direct`/`code_tools` chains.
- `run_turn(model=..., effort=...)` puts the choice, `approve` and `cancel` into the run config.
- Web: per-session model, `POST /api/sessions/<id>/model`, `GET /api/models`, handshake
  `features.models`; Claude's tool use appears as tool cards; no `PROTOCOL` bump (additive).

## Left to the builder

Recap length (keep it short: the messages since the last Claude Code answer, capped), the exact
privacy wording, layout of the picker within the existing header.
