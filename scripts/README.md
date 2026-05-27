# scripts/

Operator-authored manual smoke probes for new LLM agents — **not part of the pytest test suite**. Each `smoke_<agent>.py` is a one-shot script that invokes a real Anthropic API call (Opus / Sonnet / Haiku) on a hand-crafted state and prints the verdict. Use after a new agent module lands, before tagging the stage that introduced it. The CI gate runs the stubbed-agent unit tests; these are the human-in-the-loop sanity check the stubs cannot replace.

**Windows event-loop convention:** every async script in `scripts/` MUST set `asyncio.WindowsSelectorEventLoopPolicy()` before any `psycopg` / `langgraph` import (Python 3.13 on Windows defaults to `ProactorEventLoop`, which is incompatible with the project's psycopg-async stack — SPEC §6 change log 2026-05-27 Stage 3c). Pytest-driven tests inherit the same policy via pytest-asyncio's Selector default; only standalone scripts need the explicit set.
