# scripts/

Operator-authored manual smoke probes — **not part of the pytest test suite**. Use after a new agent module lands (before tagging the stage that introduced it) or during dashboard iteration. The CI gate runs the stubbed-agent unit tests; these scripts are the human-in-the-loop sanity checks the stubs cannot replace.

**Windows event-loop convention:** every async script in `scripts/` MUST set `asyncio.WindowsSelectorEventLoopPolicy()` before any `psycopg` / `langgraph` import (Python 3.13 on Windows defaults to `ProactorEventLoop`, which is incompatible with the project's psycopg-async stack — SPEC §6 change log 2026-05-27 Stage 3c). Pytest-driven tests inherit the same policy via pytest-asyncio's Selector default; only standalone scripts need the explicit set.

## Available scripts

| Script | Purpose | Stage |
|---|---|---|
| `smoke_critic.py` | Manual Opus probe of the critic agent against a default-hugging strategy file. One Opus 4.7 call (~$0.10–$0.30). | Stage 5 |
| `smoke_researcher.py` | Manual Sonnet/Opus probe of the full research subgraph (load_context → researcher → generator → critic → revise_or_proceed → lookahead_gate). | Stage 5 |
| `smoke_risk_analyst.py` | Manual Opus probe of `risk_analyst_node` against a hand-crafted robustness summary. One Opus 4.7 call. | Stage 4 |
| `seed_dashboard_data.py` | Operator-run seed for dashboard development. Three preset shapes (`clean_pass`, `marginal`, `kill_switch`) selectable via `--shape`. Parks a checkpoint at the appropriate HITL interrupt + inserts a `strategy_registry` row. Permanent (vs the throwaway `midstage_seed.py` convention). | Stage 6 |
