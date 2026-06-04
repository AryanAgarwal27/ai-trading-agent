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

## Development setup

Dependencies are locked with [uv](https://docs.astral.sh/uv/) (`uv.lock`, universal across Python 3.11–3.13) so local and CI resolve **identical** versions — this is what closes the type-stub drift that kept mypy non-blocking through Stage 9 (SPEC change-log 2026-05-28).

```sh
# One-time: install uv (https://docs.astral.sh/uv/getting-started/installation/)
# Then create/refresh the local .venv from the lock (project + dev tools):
uv sync --locked            # exact locked versions; errors if the lock is stale

# Run any tool inside the synced env:
uv run pytest -m "not integration" tests/unit
uv run mypy orchestrator tests
uv run ruff check .
```

To **add or change a dependency**: edit `pyproject.toml` (runtime deps under `[project].dependencies`; dev tools under `[dependency-groups].dev`), then re-lock and re-sync:

```sh
uv lock                     # re-resolve; updates uv.lock
uv sync --locked            # install the new lock
```

Commit `pyproject.toml` **and** `uv.lock` together. CI runs `uv lock --locked` as a drift guard, so a pyproject edit pushed without a matching re-lock fails the build.

> The bare `python` on PATH is system 3.10 without project deps (SPEC 2026-05-27). Inside an activated venv use `python`; in non-interactive/agent shells use the explicit interpreter `.venv\Scripts\python.exe -m <module>`, or prefer `uv run <tool>` which always targets the synced env.

## Resetting local infrastructure

`docker compose down -v` wipes the named Postgres volume (`ait_postgres_data`). The init script in [db/init/01_create_databases.sql](db/init/01_create_databases.sql) re-creates the three logical DBs + pgvector on the next `docker compose up`, but the `app` schema (the five tables from BRD §5.8) is Alembic-owned and **must be re-applied manually**:

```sh
.\.venv\Scripts\alembic.exe upgrade head
```

The LangGraph saver/store tables (`checkpoints*`, `store*`) are recreated automatically by the FastAPI lifespan calling `checkpointer.setup()` / `store.setup()` on next app start.

## License

Private; all rights reserved. See [`LICENSE`](LICENSE).
