"""Stage 9 DoD end-to-end (BRD §13 row 9): "runs nightly + on every thread
completion; spawns up to capacity; logs decisions."

Three scenarios, each against REAL Postgres + REAL Redis + the REAL parent graph
and REAL telemetry writer (the agent is stubbed deterministic — no live LLM;
that is the operator smoke, scripts/smoke_supervisor.py):

  1. NIGHTLY CRON path — fire scheduler.supervisor_cron_job directly (the cron
     trigger fires this; we invoke it, not the wall-clock schedule). Proves the
     cron path runs run_supervisor at trigger="cron" and lands a committed
     supervisor_decision telemetry row.
  2. EVENT COMPLETION path — publish a thread_completed event to Redis; the 9e
     subscription debounces it into exactly ONE coalesced run at trigger="event".
  3. SPAWN-UP-TO-CAPACITY path — seed 2 active strategies (headroom = 2 against
     MAX_CONCURRENT_STRATEGIES=4); the agent decides THREE spawns; exactly 2
     execute, the 3rd is refused (capacity_exceeded), and the telemetry audit
     records all three decisions + per-action outcomes.

Marked @pytest.mark.integration → runs in the Stage 9g CI job (postgres + redis
service containers). Skips cleanly if the DB env is unset.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from functools import partial
from typing import Any

import psycopg
import pytest
import redis.asyncio as aioredis
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from orchestrator.graph import build_per_strategy_graph
from orchestrator.observability import events
from orchestrator.observability.events import publish_thread_completed
from orchestrator.scheduler import SUPERVISOR_JOBSTORE, supervisor_cron_job
from orchestrator.supervisor import SupervisorAction, SupervisorDecision, run_supervisor
from orchestrator.supervisor_subscription import (
    cancel_supervisor_subscription,
    make_schedule_supervisor_run,
    run_supervisor_subscription,
)

pytestmark = pytest.mark.integration

load_dotenv()

_REQUIRED_ENV = ("DATABASE_URL", "LANGGRAPH_CHECKPOINT_URI", "LANGGRAPH_STORE_URI")
_EVENT_DEBOUNCE_S = 2.0


def _skip_if_no_db() -> None:
    missing = [v for v in _REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        pytest.skip(f"integration DB env not set: {missing}")


class _StubAgent:
    """Deterministic agent — returns a fixed SupervisorDecision (no live LLM)."""

    def __init__(self, decision: SupervisorDecision) -> None:
        self._decision = decision

    async def ainvoke(self, payload: Any, config: Any = None) -> dict[str, Any]:
        return {"structured_response": self._decision}


def _no_op_decision(rationale: str) -> SupervisorDecision:
    return SupervisorDecision(actions=[], overall_rationale=rationale, confidence=0.5)


async def _app_conn() -> psycopg.AsyncConnection:
    return await psycopg.AsyncConnection.connect(events._libpq_dsn(os.environ["DATABASE_URL"]))


async def _cleanup(conn: psycopg.AsyncConnection, strategy_ids: list[str]) -> None:
    async with conn.cursor() as cur:
        await cur.execute("DELETE FROM telemetry WHERE source = 'supervisor_decision'")
        for sid in strategy_ids:
            await cur.execute("DELETE FROM telemetry WHERE strategy_id = %s", (sid,))
            await cur.execute("DELETE FROM gate_audits WHERE strategy_id = %s", (sid,))
            await cur.execute("DELETE FROM strategy_registry WHERE strategy_id = %s", (sid,))
    await conn.commit()


async def _seed_registry_row(conn: psycopg.AsyncConnection, strategy_id: str, stage: str) -> None:
    """Insert a committed non-archived registry row so it counts toward `active`."""
    async with conn.cursor() as cur:
        await cur.execute(
            """
            INSERT INTO strategy_registry
              (strategy_id, thread_id, name, template, stage, pairs, timeframe,
               started_at, last_updated)
            VALUES (%s, %s, %s, 'pending', %s, %s, '5m', now(), now())
            """,
            (strategy_id, f"strategy_{strategy_id}", f"seed-{strategy_id[:6]}", stage, "[]"),
        )
    await conn.commit()


async def _supervisor_telemetry_rows(
    conn: psycopg.AsyncConnection, *, trigger: str | None = None
) -> list[dict[str, Any]]:
    sql = (
        "SELECT metrics FROM telemetry WHERE source = 'supervisor_decision' "
        "ORDER BY snapshot_at DESC"
    )
    async with conn.cursor() as cur:
        await cur.execute(sql)
        rows = await cur.fetchall()
    metrics = [r[0] for r in rows]
    if trigger is not None:
        metrics = [m for m in metrics if m.get("trigger") == trigger]
    return metrics


# ════════════════════════════════════════════════════════════════════════
# 1. Nightly cron path
# ════════════════════════════════════════════════════════════════════════


async def test_nightly_cron_path_runs_supervisor_and_logs_decision() -> None:
    """supervisor_cron_job (the cron trigger's target) → run_supervisor at
    trigger="cron" → committed supervisor_decision telemetry row."""
    _skip_if_no_db()
    async with (
        AsyncPostgresSaver.from_conn_string(os.environ["LANGGRAPH_CHECKPOINT_URI"]) as saver,
        AsyncPostgresStore.from_conn_string(os.environ["LANGGRAPH_STORE_URI"]) as store,
    ):
        await saver.setup()
        await store.setup()
        graph = build_per_strategy_graph(saver, store)

        cleanup_conn = await _app_conn()
        try:
            await _cleanup(cleanup_conn, [])  # clear any stale supervisor rows

            # The cron job opens its OWN conn internally; bind the agent + a no-op
            # spawn recorder into the runner so no live LLM / real research kick.
            run_fn = partial(
                run_supervisor,
                graph,
                store,
                agent=_StubAgent(_no_op_decision("nightly: let threads run")),
                spawn_thread_fn=_record_noop,
            )
            await supervisor_cron_job(run_supervisor_fn=run_fn)

            rows = await _supervisor_telemetry_rows(cleanup_conn, trigger="cron")
            assert len(rows) == 1, "cron run must write exactly one supervisor_decision row"
            assert rows[0]["trigger"] == "cron"
            assert rows[0]["flake"] is False
            assert "decision" in rows[0] and "action_results" in rows[0]
        finally:
            await _cleanup(cleanup_conn, [])
            await cleanup_conn.close()


async def _record_noop(strategy_id: str) -> None:
    return None


# ════════════════════════════════════════════════════════════════════════
# 2. Event completion path
# ════════════════════════════════════════════════════════════════════════


def _started_scheduler() -> AsyncIOScheduler:
    sched = AsyncIOScheduler(jobstores={"default": MemoryJobStore()}, timezone="UTC")
    sched.add_jobstore(MemoryJobStore(), SUPERVISOR_JOBSTORE)
    sched.start()
    return sched


async def test_event_completion_path_triggers_one_run() -> None:
    """A published thread_completed event → the 9e subscription debounces to
    exactly ONE run at trigger="event" → committed telemetry row."""
    _skip_if_no_db()
    redis_url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
    redis_client = aioredis.from_url(redis_url)  # type: ignore[no-untyped-call]

    async with (
        AsyncPostgresSaver.from_conn_string(os.environ["LANGGRAPH_CHECKPOINT_URI"]) as saver,
        AsyncPostgresStore.from_conn_string(os.environ["LANGGRAPH_STORE_URI"]) as store,
    ):
        await saver.setup()
        await store.setup()
        graph = build_per_strategy_graph(saver, store)

        cleanup_conn = await _app_conn()
        run_fn = partial(
            run_supervisor,
            graph,
            store,
            agent=_StubAgent(_no_op_decision("event: portfolio steady")),
            spawn_thread_fn=_record_noop,
        )
        scheduler = _started_scheduler()
        schedule_fn = make_schedule_supervisor_run(
            scheduler, run_fn, debounce_seconds=_EVENT_DEBOUNCE_S
        )
        sub_task = asyncio.create_task(
            run_supervisor_subscription(redis_client, schedule_supervisor_run_fn=schedule_fn)
        )
        try:
            await _cleanup(cleanup_conn, [])
            await asyncio.sleep(1.0)  # let PSUBSCRIBE land (Redis drops pre-subscribe msgs)

            await publish_thread_completed(
                "e2e-evt",
                {
                    "strategy_id": "e2e-evt",
                    "completed_at": "2026-06-04T03:00:00+00:00",
                    "final_stage": "archived",
                    "completion_reason": "retired_by_supervisor",
                },
            )
            await asyncio.sleep(_EVENT_DEBOUNCE_S + 5.0)  # debounce + one-shot fire margin

            rows = await _supervisor_telemetry_rows(cleanup_conn, trigger="event")
            assert len(rows) == 1, f"expected exactly one coalesced event run, got {len(rows)}"
            assert rows[0]["trigger"] == "event"
        finally:
            await cancel_supervisor_subscription(sub_task)
            scheduler.shutdown(wait=False)
            await redis_client.aclose()
            await _cleanup(cleanup_conn, [])
            await cleanup_conn.close()


# ════════════════════════════════════════════════════════════════════════
# 3. Spawn-up-to-capacity path
# ════════════════════════════════════════════════════════════════════════


async def test_spawn_up_to_capacity_refuses_overflow_and_audits_all() -> None:
    """Seed 2 active (headroom 2 under MAX_CONCURRENT_STRATEGIES=4); agent
    decides 3 spawns → 2 execute, 1 refused (capacity_exceeded), telemetry audit
    records ALL THREE decisions + per-action outcomes."""
    _skip_if_no_db()
    seed_ids = [f"e2e-seed-{uuid.uuid4().hex[:8]}" for _ in range(2)]
    spawned_ids: list[str] = []

    async def _recorder(strategy_id: str) -> None:
        spawned_ids.append(strategy_id)

    decision = SupervisorDecision(
        actions=[
            SupervisorAction(action="spawn", name="cap_a", rationale="slot 1 free"),
            SupervisorAction(action="spawn", name="cap_b", rationale="slot 2 free"),
            SupervisorAction(
                action="spawn", name="cap_c", rationale="over capacity — should refuse"
            ),
        ],
        overall_rationale="fill the two free slots; the third tests the gate",
        confidence=0.7,
    )

    async with (
        AsyncPostgresSaver.from_conn_string(os.environ["LANGGRAPH_CHECKPOINT_URI"]) as saver,
        AsyncPostgresStore.from_conn_string(os.environ["LANGGRAPH_STORE_URI"]) as store,
    ):
        await saver.setup()
        await store.setup()
        graph = build_per_strategy_graph(saver, store)

        conn = await _app_conn()
        try:
            await _cleanup(conn, [])
            # Seed 2 non-archived strategies → active=2, headroom=2.
            await _seed_registry_row(conn, seed_ids[0], "paper")
            await _seed_registry_row(conn, seed_ids[1], "research")

            await run_supervisor(
                graph,
                store,
                conn,
                trigger="cron",
                agent=_StubAgent(decision),
                spawn_thread_fn=_recorder,
            )

            # Exactly 2 spawns executed (the 2 free slots); the 3rd was refused.
            assert len(spawned_ids) == 2, f"expected 2 spawns, got {len(spawned_ids)}"

            # Registry: 2 seeded + 2 spawned research rows = 4 non-archived.
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT count(*) FROM strategy_registry WHERE stage != 'archived'"
                )
                active_row = await cur.fetchone()
                assert active_row is not None
                active = active_row[0]
            assert active == 4, f"capacity gate must cap active at 4, got {active}"

            # Telemetry audit (Fork-2 guard): records all THREE decisions + the
            # per-action outcomes, incl. WHICH spawn the gate refused.
            rows = await _supervisor_telemetry_rows(conn, trigger="cron")
            assert len(rows) == 1
            metrics = rows[0]
            assert len(metrics["decision"]["actions"]) == 3, "audit must show all 3 decided spawns"
            results = [r["result"] for r in metrics["action_results"]]
            spawned_flags = [r.get("spawned") for r in results]
            assert spawned_flags.count(True) == 2, "exactly 2 spawns executed"
            assert spawned_flags.count(False) == 1, "exactly 1 spawn refused"
            refused = next(r for r in results if r.get("spawned") is False)
            assert refused["reason"] == "capacity_exceeded"
            assert refused["limit"] == 4
        finally:
            await _cleanup(conn, seed_ids + spawned_ids)
            await conn.close()
