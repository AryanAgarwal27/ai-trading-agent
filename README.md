# ai-trading-agent

LangGraph-orchestrated autonomous agent that proposes, validates, paper-trades, and (with human approval) live-trades crypto-spot strategies. Freqtrade is the execution layer; FreqAI is its optional ML prediction layer; LangGraph is the brain.

**Single operator, $500 starting capital, paper-trade ≥30 days before any live promotion, human-in-the-loop on every paper→live and live→pause transition.**

---

## Read these before anything else

- [`BRD.md`](BRD.md) — the project contract and single source of truth. Read end-to-end at the start of every session.
- [`SPEC.md`](SPEC.md) — operator decisions that fill in BRD §18 blanks (exchange, pairs, capital cap, backup target, etc.). Read after BRD.

`BRD.md` overrides anything else; `SPEC.md` overrides casual chat preferences that conflict with §1–§4 of itself. See `SPEC.md §4.4` for the session protocol that binds Claude Code on every session.

## Current stage

See `BRD.md §13` for the full stage table. Locate the current stage with:

```sh
git log --oneline -20
git tag --list 'stage-*-complete'
```

## Resetting local infrastructure

`docker compose down -v` wipes the named Postgres volume (`ait_postgres_data`). The init script in [db/init/01_create_databases.sql](db/init/01_create_databases.sql) re-creates the three logical DBs + pgvector on the next `docker compose up`, but the `app` schema (the five tables from BRD §5.8) is Alembic-owned and **must be re-applied manually**:

```sh
.\.venv\Scripts\alembic.exe upgrade head
```

The LangGraph saver/store tables (`checkpoints*`, `store*`) are recreated automatically by the FastAPI lifespan calling `checkpointer.setup()` / `store.setup()` on next app start.

## License

Private; all rights reserved. See [`LICENSE`](LICENSE).
