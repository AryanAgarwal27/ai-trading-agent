"""Seed the app DB + Store with realistic data for the 9c supervisor smoke.

Idempotently inserts material the real Sonnet supervisor can reason over so
``scripts/smoke_supervisor.py`` exercises the full protocol (capacity check,
failure/win citation, stall-retire signal) rather than deciding against an
empty portfolio:

  - 3 ``strategy_registry`` rows (all non-archived → active=3, headroom=1):
      * seed_research_a        — research,    1 day old
      * seed_paper_gate_stalled— paper_gate, 18 days at the gate (retire candidate)
      * seed_paper_healthy     — paper,       8 days (mid-run, do NOT retire)
  - 1 ``regime_log`` row: regime="mid_vol_flat" at now() (→ the current regime)
  - 2 Store entries under the CURRENT regime so query_store returns specifics:
      * ("failures", "mid_vol_flat") — an EMA-crossover loss with named params
      * ("wins",     "mid_vol_flat") — a tight-Bollinger mean-reversion win

Schema mapping (BRD §5.8): your spec's ``created_at`` → ``started_at``;
``recorded_at`` → ``regime_log.at``. To drive the stall signal, each row's
``last_updated`` (the ``last_transition_at`` proxy ``alist_strategies`` exposes)
is set to the same age as ``started_at``.

Idempotent: registry rows upsert by ``strategy_id`` (ON CONFLICT DO UPDATE);
the regime_log seed row is delete-then-insert by ``detector="seed_smoke"``;
Store entries overwrite by key. Re-running leaves a clean seeded state and never
touches non-seed rows. Reset with ``scripts/clear_supervisor_smoke.py``.

Run from the repo root::

    .venv/Scripts/python.exe scripts/seed_supervisor_smoke.py

Requires DATABASE_URL + LANGGRAPH_STORE_URI in .env and a migrated app DB.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json  # noqa: E402
import os  # noqa: E402
from datetime import UTC, datetime, timedelta  # noqa: E402

import psycopg  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from langgraph.store.postgres.aio import AsyncPostgresStore  # noqa: E402

from orchestrator.observability import events  # noqa: E402
from orchestrator.tools.store_queries import aput_failure, aput_win  # noqa: E402

SEED_REGIME = "mid_vol_flat"
SEED_DETECTOR = "seed_smoke"

# (strategy_id, name, template, stage, pairs, timeframe, age_days)
_REGISTRY_SEEDS = [
    ("seed_research_a", "ema_crossover_explore", "pending", "research", ["BTC/USDT"], "5m", 1),
    (
        "seed_paper_gate_stalled",
        "bb_breakout_v2",
        "bb_breakout",
        "paper_gate",
        ["ETH/USDT"],
        "15m",
        18,
    ),
    (
        "seed_paper_healthy",
        "mean_reversion_4h",
        "mean_reversion",
        "paper",
        ["BTC/USDT", "SOL/USDT"],
        "4h",
        8,
    ),
]

_FAILURE_KEY = "seed_fail_ema_btc"
_FAILURE_PAYLOAD = {
    "hypothesis": "EMA crossover 20/50 on BTC/USDT 5m",
    "template": "ema_crossover",
    "result": "lost -4.2% on whipsaws during chop",
    "params": {"fast": 20, "slow": 50, "bb_std": 2.0},
}

_WIN_KEY = "seed_win_meanrev_eth"
_WIN_PAYLOAD = {
    "hypothesis": "Mean-reversion with tight Bollinger on ETH 15m",
    "template": "mean_reversion_tight_bb",
    "result": "+8.1% over paper test",
    "params": {"bb_std": 2.4, "rsi_oversold": 25},
}


async def main() -> None:
    for var in ("DATABASE_URL", "LANGGRAPH_STORE_URI"):
        if not os.environ.get(var):
            print(f"[seed] MISSING ENV: {var} — aborting.")
            return

    now = datetime.now(UTC)
    conn = await psycopg.AsyncConnection.connect(events._libpq_dsn(os.environ["DATABASE_URL"]))
    try:
        async with conn.cursor() as cur:
            for sid, name, template, stage, pairs, tf, age_days in _REGISTRY_SEEDS:
                ts = now - timedelta(days=age_days)
                await cur.execute(
                    """
                    INSERT INTO strategy_registry
                      (strategy_id, thread_id, name, template, stage, pairs,
                       timeframe, started_at, last_updated)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (strategy_id) DO UPDATE SET
                      thread_id = EXCLUDED.thread_id,
                      name = EXCLUDED.name,
                      template = EXCLUDED.template,
                      stage = EXCLUDED.stage,
                      pairs = EXCLUDED.pairs,
                      timeframe = EXCLUDED.timeframe,
                      started_at = EXCLUDED.started_at,
                      last_updated = EXCLUDED.last_updated,
                      failure_reason = NULL
                    """,
                    (sid, f"strategy_{sid}", name, template, stage, json.dumps(pairs), tf, ts, ts),
                )

            # regime_log: delete-then-insert the seed row so re-runs stay idempotent
            # (the (at, detector) PK would otherwise accumulate one row per run).
            await cur.execute("DELETE FROM regime_log WHERE detector = %s", (SEED_DETECTOR,))
            await cur.execute(
                "INSERT INTO regime_log (at, regime, features, detector) "
                "VALUES (now(), %s, %s, %s)",
                (SEED_REGIME, json.dumps({"note": "supervisor smoke seed"}), SEED_DETECTOR),
            )
        await conn.commit()
    finally:
        await conn.close()

    async with AsyncPostgresStore.from_conn_string(os.environ["LANGGRAPH_STORE_URI"]) as store:
        await store.setup()
        await aput_failure(
            store, regime=SEED_REGIME, strategy_id=_FAILURE_KEY, payload=_FAILURE_PAYLOAD
        )
        await aput_win(store, regime=SEED_REGIME, strategy_id=_WIN_KEY, payload=_WIN_PAYLOAD)

    print("[seed] OK — seeded for the supervisor smoke:")
    print(f"[seed]   registry: {[s[0] for s in _REGISTRY_SEEDS]} (active=3, headroom=1)")
    print(f"[seed]   regime_log: {SEED_REGIME} (detector={SEED_DETECTOR})")
    print(
        f"[seed]   store: ('failures','{SEED_REGIME}')/{_FAILURE_KEY}, "
        f"('wins','{SEED_REGIME}')/{_WIN_KEY}"
    )
    print("[seed] Next: .venv/Scripts/python.exe scripts/smoke_supervisor.py")


if __name__ == "__main__":
    asyncio.run(main())
