# Long-term memory (`memories/`)

- `table.py`: 3 tables (episodic, semantic, procedural), `EMBEDDING_DIM = 384`, HNSW cosine indexes.
- `store.py`: persistence + search. `add_fact` overwrites a near-duplicate with the newest wording, because a fact and its negation embed almost identically. Fact search is hybrid (`rank_facts`: vector + BM25 from `retrieval.py`); `keywords=False` is vector only (used inside `remember`, so word matches do not send every save to the review model). `profile_facts`: `PROFILE_CATEGORIES`, newest first.
- `extractor.py`: LLM structured extraction of one turn at temperature 0 (the chat default made facts come and go, and once stored a raw key); skips greeting/ack-only turns (`TRIVIAL_WORDS`); fixed `FactCategory` (unknown -> `general`); may mark known facts superseded. `find_superseded` reviews stored facts near the new ones.
- `manager.py`: `recall(query, context)` = profile facts + search + a few follow-up hits (`memory_query` builds both strings from the messages; `graph/node.py` uses it). `remember` drops secrets (`looks_secret`), deletes superseded facts (from the extractor, and from the review of facts within `REVIEW_DISTANCE`), stores the rest.
- Eval: `$env:RUN_MEMORY_EVAL = "1"; python -m unittest tests.test_memory_eval -v` (real embeddings and DB, rolled back; 33/35). Raise `MIN_HIT_RATE` when retrieval improves; a change that lowers it is a regression.

Memory is optional: if Postgres is unreachable, `dependency["memory"]` is `None`, a warning is logged, and the app runs without it. Distance thresholds in `store.py` were calibrated against bge-small; re-check them if the embedding model changes. Job notices are never recalled for or learned from (root CLAUDE.md, Hardening).

- `diagnose.py` (pure: `parse_netstat`, `classify`, `redact_url`) and `check.py` (`python -m memories.check`, `explain_failure`) explain connection failures: a foreign process on the DB port gives `WRONG_SERVER`. `dependency.memory_init` logs the same hint. The container is published on host port 15432 (`compose.yaml`); a foreign listener next to Docker on the port still counts as `WRONG_SERVER`.
