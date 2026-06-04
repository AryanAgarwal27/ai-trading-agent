"""Operator-run smoke probe: real Sonnet 4.6 supervisor decision (Stage 9c).

Drives the REAL production reasoning path via ``run_supervisor(dry_run=True)``
— NOT a hand-built kickoff + direct ``agent.ainvoke`` (the original probe did
that and bypassed the runner entirely, which is exactly why the first smoke
showed the agent skipping read-tool calls: it was testing the wrong code path).
With ``dry_run=True`` the runner executes sync → snapshot → regime → strategies
→ ContextVars → agent → decision, then SKIPS all writes and ``conn.rollback()``s
— so this exercises the exact sequence the nightly cron will run (the agent
calls view_portfolio / get_market_regime / query_store / list_strategies as its
protocol mandates) without mutating the DB.

Run from the repo root::

    .venv/Scripts/python.exe scripts/smoke_supervisor.py

Requires ANTHROPIC_API_KEY + DATABASE_URL + LANGGRAPH_CHECKPOINT_URI +
LANGGRAPH_STORE_URI in .env and a migrated app DB. Seed first with
scripts/seed_supervisor_smoke.py; reset with scripts/clear_supervisor_smoke.py.
Enable LANGSMITH_TRACING=true to see the read-tool call sequence in LangSmith.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402

# Windows event-loop convention — MUST precede any psycopg / langgraph import.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os  # noqa: E402
import time  # noqa: E402

import psycopg  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: E402
from langgraph.store.postgres.aio import AsyncPostgresStore  # noqa: E402

from orchestrator.graph import build_per_strategy_graph  # noqa: E402
from orchestrator.observability import events  # noqa: E402
from orchestrator.supervisor import (  # noqa: E402
    aget_current_regime,
    aget_portfolio_snapshot,
    alist_strategies,
    run_supervisor,
)


async def main() -> None:
    required = (
        "ANTHROPIC_API_KEY",
        "DATABASE_URL",
        "LANGGRAPH_CHECKPOINT_URI",
        "LANGGRAPH_STORE_URI",
    )
    for var in required:
        if not os.environ.get(var):
            print(f"[smoke_supervisor] MISSING ENV: {var} — aborting.")
            return

    conn = await psycopg.AsyncConnection.connect(events._libpq_dsn(os.environ["DATABASE_URL"]))
    try:
        # The "before" view — what the portfolio looks like going in (the runner
        # re-reads these internally; printed here so the operator can compare the
        # input state to the agent's decision).
        snapshot = await aget_portfolio_snapshot(conn)
        regime = await aget_current_regime(conn)
        strategies = await alist_strategies(conn)

        print("[smoke_supervisor] dry_run=True — production reasoning path, NO DB mutation.")
        print(f"[smoke_supervisor] portfolio snapshot: {snapshot}")
        print(f"[smoke_supervisor] current regime: {regime}")
        print(f"[smoke_supervisor] active strategies: {len(strategies)}")
        for s in strategies:
            print(f"    - {s['strategy_id']} stage={s['stage']} age_days={s.get('age_days')}")
        print("[smoke_supervisor] running run_supervisor(dry_run=True) — real Sonnet 4.6...")

        async with (
            AsyncPostgresSaver.from_conn_string(os.environ["LANGGRAPH_CHECKPOINT_URI"]) as saver,
            AsyncPostgresStore.from_conn_string(os.environ["LANGGRAPH_STORE_URI"]) as store,
        ):
            await saver.setup()
            await store.setup()
            graph = build_per_strategy_graph(saver, store)

            t0 = time.perf_counter()
            decision = await run_supervisor(
                graph=graph,
                store=store,
                conn=conn,
                trigger="manual",
                dry_run=True,
            )
            wall = time.perf_counter() - t0
    finally:
        await conn.close()

    print("\n" + "=" * 72)
    print(" 9c SUPERVISOR DECISION (real Sonnet 4.6, via run_supervisor dry_run)")
    print("=" * 72)
    print(f"\nWall-clock: {wall:.2f} s")
    print(f"\noverall_rationale: {decision.overall_rationale}")
    print(f"confidence: {decision.confidence}")
    print(f"\nactions ({len(decision.actions)}):")
    if not decision.actions:
        print("    (none — the agent chose to let existing threads run)")
    for i, a in enumerate(decision.actions, start=1):
        print(f"  [{i}] {a.action}")
        if a.action == "spawn":
            print(f"      name={a.name} template={a.template} pairs={a.pairs} tf={a.timeframe}")
        elif a.action == "retire":
            print(f"      strategy_id={a.strategy_id}")
        print(f"      rationale: {a.rationale}")
    print("\n" + "=" * 72)
    print(" RESULT: decision printed; dry_run rolled back (nothing executed/committed).")
    print(" Check LangSmith for the tool-call trace — expect view_portfolio,")
    print(" get_market_regime, query_store(failures), query_store(wins), and")
    print(" list_strategies all firing before SupervisorDecision.")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
