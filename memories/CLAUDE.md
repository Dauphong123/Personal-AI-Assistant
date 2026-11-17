# Long-term memory (`memories/`)

- `table.py`: 3 tables (episodic, semantic, procedural), `EMBEDDING_DIM = 384`, HNSW cosine indexes.
- `store.py`: persistence + vector search. `add_fact` overwrites a near-duplicate with the newest wording, because a fact and its negation embed almost identically.
- `extractor.py`: LLM structured extraction of one turn; skips trivial turns; may mark known facts superseded.
- `manager.py`: `recall` formats memories into the system prompt; `remember` stores a turn.

Memory is optional: if Postgres is unreachable, `dependency["memory"]` is `None`, a warning is logged, and the app runs without it. Distance thresholds in `store.py` were calibrated against bge-small; re-check them if the embedding model changes. Job notices are never recalled for or learned from (root CLAUDE.md, Hardening).

- `diagnose.py` (pure: `parse_netstat`, `classify`, `redact_url`) and `check.py` (`python -m memories.check`, `explain_failure`) explain connection failures: a foreign process on the DB port gives `WRONG_SERVER`. `dependency.memory_init` logs the same hint. The container is published on host port 15432 (`compose.yaml`); a foreign listener next to Docker on the port still counts as `WRONG_SERVER`.
