"""D-11 integration tests — the supervisor serialization advisory lock.

Against REAL Postgres: prove the ``pg_advisory_xact_lock`` around
``run_supervisor`` (a) is genuinely mutually-exclusive, (b) auto-releases when
the holding connection closes (crash-safety), and (c) serializes two concurrent
``run_supervisor`` invocations — the second BLOCKS on the lock rather than
racing the capacity read-then-write across the distinct cron/event/manual job
ids (the D-11 overshoot the deferral describes).

Marked ``integration`` (real app DB), NOT ``freqtrade`` (no Docker).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import psycopg
import pytest
from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

import orchestrator.observability.events as events
from orchestrator.graph import build_per_strategy_graph
from orchestrator.supervisor import (
    _SUPERVISOR_ADVISORY_LOCK_KEY,
    SupervisorDecision,
    run_supervisor,
)

pytestmark = pytest.mark.integration

load_dotenv()

_REQUIRED_ENV = ("DATABASE_URL", "LANGGRAPH_CHECKPOINT_URI", "LANGGRAPH_STORE_URI")


def _skip_if_no_db() -> None:
    missing = [v for v in _REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        pytest.skip(f"integration DB env not set: {missing}")


async def _app_conn() -> psycopg.AsyncConnection:
    return await psycopg.AsyncConnection.connect(events._libpq_dsn(os.environ["DATABASE_URL"]))


async def _try_lock(conn: psycopg.AsyncConnection) -> bool:
    """pg_try_advisory_xact_lock(KEY) on conn — True iff it acquired the lock."""
    async with conn.cursor() as cur:
        await cur.execute("SELECT pg_try_advisory_xact_lock(%s)", (_SUPERVISOR_ADVISORY_LOCK_KEY,))
        row = await cur.fetchone()
    assert row is not None
    return bool(row[0])


async def test_advisory_lock_is_exclusive_and_releases_on_conn_close() -> None:
    """conn1 holds the xact lock → conn2 cannot acquire it; close conn1 → the lock
    auto-releases (xact-scoped) → conn2 acquires it. The crash-safety property:
    a dead holder's connection closing frees the next run, no leaked lock."""
    _skip_if_no_db()

    conn1 = await _app_conn()
    conn2 = await _app_conn()
    try:
        # conn1 takes the lock inside its (uncommitted) transaction.
        async with conn1.cursor() as cur:
            await cur.execute("SELECT pg_advisory_xact_lock(%s)", (_SUPERVISOR_ADVISORY_LOCK_KEY,))

        # conn2 is refused while conn1 holds it.
        assert await _try_lock(conn2) is False

        # conn1 closes → its transaction ends → the xact lock releases.
        await conn1.close()

        # conn2 can now take it (same txn; the prior try did not acquire).
        assert await _try_lock(conn2) is True
    finally:
        await conn2.rollback()
        if not conn1.closed:
            await conn1.close()
        await conn2.close()


async def test_concurrent_run_supervisor_serializes_on_lock() -> None:
    """A second run_supervisor BLOCKS on the advisory lock while another holds it,
    then proceeds once released — it never races the capacity read. Driven with
    dry_run=True so the blocked run does its full read path and rolls back (no
    writes to clean up); the lock is acquired regardless of dry_run."""
    _skip_if_no_db()

    decision = SupervisorDecision(actions=[], overall_rationale="noop", confidence=0.0)

    class _StubAgent:
        async def ainvoke(self, payload: Any, config: Any = None) -> dict[str, Any]:
            return {"structured_response": decision}

    async with (
        AsyncPostgresSaver.from_conn_string(os.environ["LANGGRAPH_CHECKPOINT_URI"]) as saver,
        AsyncPostgresStore.from_conn_string(os.environ["LANGGRAPH_STORE_URI"]) as store,
    ):
        await saver.setup()
        await store.setup()
        graph = build_per_strategy_graph(saver, store)

        # A holder stands in for "run A in-flight": take the lock and keep its txn.
        conn_hold = await _app_conn()
        conn_b = await _app_conn()
        task: asyncio.Task[Any] | None = None
        try:
            async with conn_hold.cursor() as cur:
                await cur.execute(
                    "SELECT pg_advisory_xact_lock(%s)", (_SUPERVISOR_ADVISORY_LOCK_KEY,)
                )

            # run B must block at _acquire_supervisor_lock — not proceed to read.
            task = asyncio.create_task(
                run_supervisor(
                    graph,
                    store,
                    conn_b,
                    trigger="event",
                    agent=_StubAgent(),
                    dry_run=True,
                )
            )
            _done, pending = await asyncio.wait({task}, timeout=1.0)
            assert task in pending, "second run_supervisor must WAIT on the advisory lock"

            # Release the holder → the blocked run acquires, reads, rolls back, ends.
            await conn_hold.close()
            result = await asyncio.wait_for(task, timeout=15)
            assert result.actions == []  # the dry_run noop decision came back
        finally:
            if task is not None and not task.done():
                task.cancel()
            if not conn_hold.closed:
                await conn_hold.close()
            await conn_b.close()
