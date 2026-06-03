"""Operator-run smoke probe: real Sonnet 4.6 supervisor decision (Stage 9c).

DECISION-ONLY by design — this hits the real Anthropic API with the real
system prompt and the ACTUAL portfolio snapshot/regime/active-strategies read
from the live app DB, prints the structured ``SupervisorDecision`` the agent
produces, and then STOPS. It does NOT execute any action and does NOT commit
anything — it never spawns a thread or mutates the registry. (The full execute
+ commit path is covered by ``run_supervisor`` and its tests; this probe exists
to let the operator eyeball the real agent's reasoning against the real
portfolio before trusting the scheduled runner.)

Run from the repo root::

    .venv/Scripts/python.exe scripts/smoke_supervisor.py

Requires ANTHROPIC_API_KEY + DATABASE_URL + LANGGRAPH_STORE_URI in .env, and a
migrated app DB (``alembic upgrade head``). The store read (query_store →
failures/wins) uses the real AsyncPostgresStore so the agent reasons over real
history; the portfolio/regime/strategies reads use a real app-DB connection.
No checkpointer is needed — the supervisor agent is stateless-per-run.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project importable when invoked directly (vs as a module).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402

# Windows event-loop convention (see scripts/README.md) — MUST precede any
# psycopg / langgraph import, including transitive ones below.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os  # noqa: E402
import time  # noqa: E402

import psycopg  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from langgraph.store.postgres.aio import AsyncPostgresStore  # noqa: E402

from orchestrator.observability import events  # noqa: E402
from orchestrator.supervisor import (  # noqa: E402
    _current_portfolio,
    _current_regime,
    _current_store,
    _current_strategies,
    aget_current_regime,
    aget_portfolio_snapshot,
    alist_strategies,
    build_supervisor_agent,
)


async def main() -> None:
    for var in ("ANTHROPIC_API_KEY", "DATABASE_URL", "LANGGRAPH_STORE_URI"):
        if not os.environ.get(var):
            print(f"[smoke_supervisor] MISSING ENV: {var} — aborting.")
            return

    conn = await psycopg.AsyncConnection.connect(events._libpq_dsn(os.environ["DATABASE_URL"]))
    try:
        snapshot = await aget_portfolio_snapshot(conn)
        regime = await aget_current_regime(conn)
        strategies = await alist_strategies(conn)
    finally:
        await conn.close()

    print("[smoke_supervisor] DECISION-ONLY — no actions executed, no DB mutation.")
    print(f"[smoke_supervisor] portfolio snapshot: {snapshot}")
    print(f"[smoke_supervisor] current regime: {regime}")
    print(f"[smoke_supervisor] active strategies: {len(strategies)}")
    for s in strategies:
        print(f"    - {s['strategy_id']} stage={s['stage']} age_days={s.get('age_days')}")
    print("[smoke_supervisor] invoking supervisor agent (real Sonnet 4.6)...")

    from langchain_core.messages import HumanMessage

    async with AsyncPostgresStore.from_conn_string(os.environ["LANGGRAPH_STORE_URI"]) as store:
        await store.setup()

        tok_store = _current_store.set(store)
        tok_regime = _current_regime.set(regime)
        tok_portfolio = _current_portfolio.set(snapshot)
        tok_strategies = _current_strategies.set(strategies)
        try:
            agent = build_supervisor_agent()
            kickoff = (
                "Run the supervisor protocol now and emit a SupervisorDecision. "
                f"Portfolio: {snapshot}. Regime: {regime}. "
                f"Active strategies: {strategies}"
            )
            t0 = time.perf_counter()
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=kickoff)]},
                config={"configurable": {"thread_id": "supervisor"}},
            )
            wall = time.perf_counter() - t0
        finally:
            _current_store.reset(tok_store)
            _current_regime.reset(tok_regime)
            _current_portfolio.reset(tok_portfolio)
            _current_strategies.reset(tok_strategies)

    decision = result.get("structured_response")

    print("\n" + "=" * 72)
    print(" 9c SUPERVISOR DECISION (real Sonnet 4.6)")
    print("=" * 72)
    print(f"\nWall-clock: {wall:.2f} s")
    if decision is None:
        print("\nNO structured_response returned — agent flake. The runner would")
        print("fall back to a no_op decision (flake=True) and still log telemetry.")
        return

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
    print(" RESULT: decision printed. Nothing executed. Check LangSmith for the")
    print(" tool-call trace (view_portfolio / query_store / get_market_regime /")
    print(" list_strategies) and token cost (project: ai-trading-agent).")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
