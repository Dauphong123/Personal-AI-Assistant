# Glossary

Domain terms for this project. Architecture words (module, interface, seam, adapter, depth, leverage, locality) follow the `codebase-design` skill.

**Turn**: one message the agent answers to completion: from a user message (or a job notice) to the final answer, including tool calls and approval pauses. Runs through `turns.run_turn`.

**Turn runner** (`turn_runner.py`, planned in F040): runs one conversation's turns one at a time on a worker thread, with stop, approval (each request has an id; an answer counts only for that request), queued job notices, an optional shared slot and job owner. Returns a typed outcome (done, stopped, too many steps, error). The CLI (`ChatLoop`) and the web (`ChatSession`) are its adapters; each keeps its own words for the outcome.

**Job notice**: a message the app sends into a conversation when a background `terminal` job ends (`jobs.finished_notice`). It is not something the user said: memory neither recalls for nor learns from it, the rule router and the Claude recap skip it. It still starts a turn of its own (own tool-call budget; summarize may split there). The rule lives in `conversation.py` (planned in F039).

**Approval**: the user's yes/no before a tool in `NEEDS_APPROVAL` (or a Claude Code edit/shell call) runs. Default no.

**Model choice**: the model a turn runs on: an Ollama name or `claude-code:<alias>` with an effort (`model_choice.py`). `MODEL` is the main model; router, summaries and memory always use it.
