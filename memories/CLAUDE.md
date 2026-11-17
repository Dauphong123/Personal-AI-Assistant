# Long-term memory (`memories/`)

- `table.py`: 3 tables (episodic, semantic, procedural), `EMBEDDING_DIM = 384`, HNSW cosine indexes.
- `store.py`: persistence + vector search. `add_fact` overwrites a near-duplicate with the newest wording, because a fact and its negation embed almost identically.
- `extractor.py`: LLM structured extraction of one turn; skips trivial turns; may mark known facts superseded.
- `manager.py`: `recall` formats memories into the system prompt; `remember` stores a turn.

Memory is optional: if Postgres is unreachable, `dependency["memory"]` is `None`, a warning is logged, and the app runs without it. Distance thresholds in `store.py` were calibrated against bge-small; re-check them if the embedding model changes. Job notices are never recalled for or learned from (root CLAUDE.md, Hardening).
