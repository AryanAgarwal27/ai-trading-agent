"""Remove ONLY the supervisor-smoke seed data (companion to seed_supervisor_smoke.py).

Lets the operator reset between smoke runs without touching any real rows.
Deletes exactly what the seed script created, by prefix / marker:

  - ``strategy_registry`` rows with ``strategy_id LIKE 'seed_%'`` (plus any
    dependent ``telemetry`` / ``gate_audits`` / ``kill_switch_events`` rows
    that FK-reference them, so the registry delete can't FK-fail).
  - ``regime_log`` rows with ``detector = 'seed_smoke'``.
  - the two seeded Store entries by key.

Non-seed rows (real strategies, real regime classifications, real telemetry —
including the portfolio-level ``source='supervisor_decision'`` rows, whose
strategy_id is NULL) are never touched.

Run from the repo root::

    .venv/Scripts/python.exe scripts/clear_supervisor_smoke.py

Requires DATABASE_URL + LANGGRAPH_STORE_URI in .env.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os  # noqa: E402

import psycopg  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from langgraph.store.postgres.aio import AsyncPostgresStore  # noqa: E402

from orchestrator.observability import events  # noqa: E402

# Keep these in lockstep with seed_supervisor_smoke.py.
SEED_REGIME = "mid_vol_flat"
SEED_DETECTOR = "seed_smoke"
_STORE_KEYS = [("failures", "seed_fail_ema_btc"), ("wins", "seed_win_meanrev_eth")]


async def main() -> None:
    for var in ("DATABASE_URL", "LANGGRAPH_STORE_URI"):
        if not os.environ.get(var):
            print(f"[clear] MISSING ENV: {var} — aborting.")
            return

    conn = await psycopg.AsyncConnection.connect(events._libpq_dsn(os.environ["DATABASE_URL"]))
    try:
        async with conn.cursor() as cur:
            # Dependent FK rows first (the smoke is decision-only, so these are
            # normally empty for seed_ ids — but delete defensively so a registry
            # delete can never FK-fail).
            await cur.execute("DELETE FROM telemetry WHERE strategy_id LIKE 'seed_%'")
            await cur.execute("DELETE FROM gate_audits WHERE strategy_id LIKE 'seed_%'")
            await cur.execute("DELETE FROM kill_switch_events WHERE strategy_id LIKE 'seed_%'")
            await cur.execute("DELETE FROM strategy_registry WHERE strategy_id LIKE 'seed_%'")
            await cur.execute("DELETE FROM regime_log WHERE detector = %s", (SEED_DETECTOR,))
        await conn.commit()
    finally:
        await conn.close()

    async with AsyncPostgresStore.from_conn_string(os.environ["LANGGRAPH_STORE_URI"]) as store:
        await store.setup()
        for category, key in _STORE_KEYS:
            await store.adelete((category, SEED_REGIME), key)

    print(
        "[clear] OK — removed supervisor-smoke seed data (seed_* registry rows, "
        f"detector={SEED_DETECTOR} regime_log, 2 store keys). Non-seed rows untouched."
    )


if __name__ == "__main__":
    asyncio.run(main())
