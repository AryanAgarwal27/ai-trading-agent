"""Integration tests for ``run_supervisor`` against real Postgres (Stage 9c).

These exercise the runner end-to-end with a REAL app-DB connection, a REAL
``AsyncPostgresSaver`` + ``AsyncPostgresStore``, and the REAL parent graph
(``build_per_strategy_graph``) — proving the true-DB fidelity the
mocked-conn unit suite (``tests/unit/test_supervisor_runner.py``) cannot:
that ``aspawn_strategy`` writes a real ``strategy_registry`` row, that
``record_telemetry`` lands a real ``telemetry`` row with
``source="supervisor_decision"``, and that the whole batch commits atomically.

The AGENT is stubbed (deterministic ``SupervisorDecision``) and
``spawn_thread_fn`` is a recorder — we do NOT kick a real LLM research run
here (that is the operator smoke, ``scripts/smoke_supervisor.py``). The
default ``audit_writer_fn`` is used so the REAL telemetry writer is tested.

Marked ``@pytest.mark.integration`` → excluded from the CI unit job (the
integration CI job lands in 9g, closing D-7). Requires Postgres on
``DATABASE_URL`` / ``LANGGRAPH_CHECKPOINT_URI`` / ``LANGGRAPH_STORE_URI`` with
the app DB migrated (``alembic upgrade head``) — same precondition as
``test_postgres_lifecycle`` / ``test_hitl_resume``. Skips cleanly if unset.
"""

from __future__ import annotations

import os
from typing import Any

import psycopg
import pytest
from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from orchestrator.graph import build_per_strategy_graph
from orchestrator.observability import events
from orchestrator.supervisor import SupervisorAction, SupervisorDecision, run_supervisor

pytestmark = pytest.mark.integration

load_dotenv()

_REQUIRED_ENV = ("DATABASE_URL", "LANGGRAPH_CHECKPOINT_URI", "LANGGRAPH_STORE_URI")


def _skip_if_no_db() -> None:
    missing = [v for v in _REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        pytest.skip(f"integration DB env not set: {missing}")


class _StubAgent:
    """Deterministic agent — returns a fixed SupervisorDecision."""

    def __init__(self, decision: SupervisorDecision) -> None:
        self._decision = decision

    async def ainvoke(self, payload: Any, config: Any = None) -> dict[str, Any]:
        return {"structured_response": self._decision}


async def _app_conn() -> psycopg.AsyncConnection:
    return await psycopg.AsyncConnection.connect(events._libpq_dsn(os.environ["DATABASE_URL"]))


async def _cleanup(conn: psycopg.AsyncConnection, strategy_ids: list[str]) -> None:
    """Remove test rows so the suite is idempotent on a shared dev DB."""
    async with conn.cursor() as cur:
        await cur.execute("DELETE FROM telemetry WHERE source = 'supervisor_decision'")
        for sid in strategy_ids:
            await cur.execute("DELETE FROM telemetry WHERE strategy_id = %s", (sid,))
            await cur.execute("DELETE FROM gate_audits WHERE strategy_id = %s", (sid,))
            await cur.execute("DELETE FROM strategy_registry WHERE strategy_id = %s", (sid,))
    await conn.commit()


async def test_spawn_writes_real_registry_row_and_telemetry() -> None:
    """Empty portfolio + spawn decision → a real research row + a real
    supervisor_decision telemetry row, committed together."""
    _skip_if_no_db()

    decision = SupervisorDecision(
        actions=[SupervisorAction(action="spawn", name="it_spawn", rationale="empty portfolio")],
        overall_rationale="seed one candidate",
        confidence=0.8,
    )

    spawned_ids: list[str] = []

    async def _recorder(sid: str) -> None:
        spawned_ids.append(sid)

    async with (
        AsyncPostgresSaver.from_conn_string(os.environ["LANGGRAPH_CHECKPOINT_URI"]) as saver,
        AsyncPostgresStore.from_conn_string(os.environ["LANGGRAPH_STORE_URI"]) as store,
    ):
        await saver.setup()
        await store.setup()
        graph = build_per_strategy_graph(saver, store)

        conn = await _app_conn()
        try:
            result = await run_supervisor(
                graph,
                store,
                conn,
                trigger="cron",
                agent=_StubAgent(decision),
                spawn_thread_fn=_recorder,  # default audit_writer → REAL telemetry write
            )
            assert result.actions[0].action == "spawn"
            assert len(spawned_ids) == 1
            sid = spawned_ids[0]

            # Real registry row at stage='research'.
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT stage, template FROM strategy_registry WHERE strategy_id = %s",
                    (sid,),
                )
                row = await cur.fetchone()
            assert row is not None
            assert row[0] == "research"

            # Real telemetry row with the supervisor decision.
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT metrics FROM telemetry WHERE source = 'supervisor_decision' "
                    "ORDER BY snapshot_at DESC LIMIT 1"
                )
                trow = await cur.fetchone()
            assert trow is not None
            metrics = trow[0]
            assert metrics["trigger"] == "cron"
            assert metrics["decision"]["actions"][0]["action"] == "spawn"
            assert metrics["action_results"][0]["result"]["spawned"] is True
        finally:
            await _cleanup(conn, spawned_ids)
            await conn.close()


# NOTE — retire end-to-end is NOT integration-tested here, deliberately.
# A faithful retire e2e needs a thread parked at a REAL interrupt (the v1
# retire contract: parked/idle threads only). Seeding that on a fresh thread
# via ``graph.aupdate_state`` raises LangGraph's
# ``InvalidUpdateError("Ambiguous update, specify as_node")`` — the same reason
# kill_subscription.py's Fork-2 guard refuses to ``aupdate_state`` a thread with
# no checkpoint. ``aretire_strategy`` itself is production-safe (its aget_state
# guard returns unknown_strategy on an empty checkpoint, so it only reaches
# aupdate_state on a real parked thread, where the update is unambiguous — the
# proven kill_subscription pattern). The retire LOGIC is fully covered by the
# unit suite (tests/unit/test_supervisor_runner.py::test_retire_* and
# tests/unit/test_supervisor_capacity.py). A real-DB retire e2e that first
# drives the graph to a genuine parked checkpoint belongs in the Stage 9h DoD
# end-to-end test, which exercises the full graph.
