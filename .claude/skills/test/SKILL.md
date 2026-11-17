---
name: test
description: Role for running and writing tests in this project and verifying a change works. Use after any code change and before saying work is done.
---

1. Run `python -m unittest discover -s tests -t .` (venv active), or the narrowest relevant module first. Postgres test skips if the DB is down; voice engine tests need `RUN_VOICE_ENGINE_TESTS=1` (downloads models).
2. Check the app still imports and initializes (importing loads models; slow is normal).
3. Test new behavior and edge cases; prefer fakes over real models/network (voice parts are injected for this).
4. Report exactly: ran, passed, failed (with output), skipped or not runnable and why.
5. Failure: switch to `debug`. Pass: switch to `docs-sync`.
