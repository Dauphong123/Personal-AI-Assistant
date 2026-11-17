---
name: architecture-review
description: Checklist for reviewing this project's architecture and technical quality. Use when asked to evaluate, audit or find weaknesses in the codebase before proposing improvements.
---

Inspect first: directory/module structure, entry points, main flow, module responsibilities, dependencies, config/env handling, DB models/connections, services/agents/nodes/tools, state and data flow, error handling/logging, tests and conventions. Trace real execution paths in source; never judge from file names.

## Architecture
- **Separation of concerns:** clear module responsibilities; business logic vs infrastructure vs config vs orchestration; functions/classes doing unrelated things.
- **Dependencies:** tight coupling, appropriate injection, initialization in the right place, circular or unused imports.
- **Lifecycle:** clear init/run/shutdown; unnecessary global state; abstractions sized to real needs.
- **Persistence:** connection/session management, transactions, cleanup, model/operation organization, persistence separated from logic.
- **Errors:** handled at proper boundaries, none silently swallowed, useful consistent logging, safe background work and tool execution.
- **Maintainability:** duplication, naming/interface consistency, needless complexity, extensibility without touching unrelated modules.

## Technical quality
Unsafe or incorrect code, runtime bugs, type errors (run Pyright), needless synchronous work, inefficient DB operations, resource leaks, security holes, missing validation, unhandled edge cases, test gaps.

Report only issues substantiated by the actual implementation. Goal: best practical architecture for the real requirements, not more patterns, classes or files.
