"""Stage 6d integration tests — FastAPI endpoints against real Postgres.

These tests drive the real FastAPI lifespan (so saver / store / Redis
client / per-thread lock map all initialize), then SWAP
``app.state.graph`` for a small test graph that has a ``paper_gate``
interrupt node — the real parent graph (built by
``build_per_strategy_graph``) has no gate nodes yet (those land in
Stage 6e + Stage 7+).

The test graph uses the same ``AsyncPostgresSaver`` the lifespan
opened, so threads we park in the test are visible to
``graph.aget_state`` through the FastAPI handlers.

All tests are marked ``@pytest.mark.integration`` — they require:
- Postgres reachable at ``DATABASE_URL`` + ``LANGGRAPH_CHECKPOINT_URI``
- The ``app`` DB migrated (BRD §5.8 schema)

Unique ``strategy_id`` / ``thread_id`` per test means no cross-test
contamination of registry / checkpoint rows. ``finally`` blocks clean
up registry + audit rows.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, TypedDict
from unittest.mock import AsyncMock

import psycopg
import pytest
from dotenv import load_dotenv
from httpx import ASGITransport, AsyncClient
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from orchestrator import main as main_module
from orchestrator.main import app
from orchestrator.observability import events

load_dotenv()

pytestmark = pytest.mark.integration

TEST_OPERATOR_TOKEN = "test-operator-token-stage6d-deadbeef"


# ─── Test graph schema + builder ───────────────────────────────────────


class _GateState(TypedDict, total=False):
    """Minimal schema for the paper_gate test graph.

    TypedDict (not bare dict) for per-key merge semantics — see the
    NOTE at the top of tests/unit/test_hitl.py for the dead end this
    avoids.
    """

    strategy_id: str
    approved: bool
    notes: str
    stage: str


def _build_paper_gate_test_graph(saver: Any) -> Any:
    """Graph: START → paper_gate(interrupt) → END.

    On resume, the gate returns ``{"approved", "notes", "stage"}`` —
    matching the contract the real Stage 6e ``paper_gate`` node will
    write. The endpoint handler reads ``post_snapshot.values.get("stage")``
    for the ``next_stage`` response field.
    """

    def paper_gate(state: _GateState) -> dict[str, Any]:
        decision = interrupt(
            {"kind": "paper_gate", "strategy_id": state.get("strategy_id")}
        )
        return {
            "approved": decision["approved"],
            "notes": decision.get("notes", ""),
            "stage": "paper" if decision["approved"] else "archived",
        }

    g: StateGraph[_GateState, _GateState, _GateState, _GateState] = StateGraph(_GateState)
    g.add_node("paper_gate", paper_gate)
    g.add_edge(START, "paper_gate")
    g.add_edge("paper_gate", END)
    return g.compile(checkpointer=saver)


# ─── App lifecycle helper ──────────────────────────────────────────────


@asynccontextmanager
async def _app_with_paper_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[AsyncClient]:
    """Drive the lifespan, swap in the paper_gate test graph, hand back
    an ``httpx.AsyncClient`` ready for requests.

    Sets ``OPERATOR_TOKEN`` in env BEFORE lifespan opens so the dependency
    sees it on the first request.
    """
    monkeypatch.setenv("OPERATOR_TOKEN", TEST_OPERATOR_TOKEN)
    async with app.router.lifespan_context(app):
        app.state.graph = _build_paper_gate_test_graph(app.state.saver)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


# ─── Postgres helpers ──────────────────────────────────────────────────


async def _open_app_conn() -> psycopg.AsyncConnection:
    return await psycopg.AsyncConnection.connect(events._libpq_dsn(os.environ["DATABASE_URL"]))


async def _seed_registry_row(strategy_id: str, thread_id: str, stage: str = "paper_gate") -> None:
    conn = await _open_app_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO strategy_registry "
                "(strategy_id, thread_id, name, template, stage, pairs, timeframe) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    strategy_id,
                    thread_id,
                    f"test-{strategy_id}",
                    "mean_reversion_template",
                    stage,
                    json.dumps(["BTC/USDT"]),
                    "5m",
                ),
            )
        await conn.commit()
    finally:
        await conn.close()


async def _cleanup_registry_and_audits(strategy_ids: list[str]) -> None:
    if not strategy_ids:
        return
    conn = await _open_app_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM gate_audits WHERE strategy_id = ANY(%s)", (strategy_ids,)
            )
            await cur.execute(
                "DELETE FROM strategy_registry WHERE strategy_id = ANY(%s)", (strategy_ids,)
            )
        await conn.commit()
    finally:
        await conn.close()


async def _park_at_paper_gate(graph: Any, strategy_id: str, thread_id: str) -> None:
    """Drive the test graph until it parks at the paper_gate interrupt."""
    config = {"configurable": {"thread_id": thread_id}}
    async for _ in graph.astream({"strategy_id": strategy_id}, config=config):
        pass


# ─── GET /threads ──────────────────────────────────────────────────────


async def test_get_threads_returns_registry_rows_with_interrupt_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two registry rows: one parked at an interrupt, one not. Both
    appear in the response with the correct ``has_pending_interrupt``
    flag."""
    parked_sid = f"sid_parked_{uuid.uuid4().hex[:8]}"
    parked_tid = f"thread_{parked_sid}"
    idle_sid = f"sid_idle_{uuid.uuid4().hex[:8]}"
    idle_tid = f"thread_{idle_sid}"

    cleanup_ids = [parked_sid, idle_sid]

    try:
        await _seed_registry_row(parked_sid, parked_tid, stage="paper_gate")
        await _seed_registry_row(idle_sid, idle_tid, stage="research")

        async with _app_with_paper_gate(monkeypatch) as client:
            # Park the first thread at the paper_gate interrupt using the
            # already-installed test graph.
            await _park_at_paper_gate(app.state.graph, parked_sid, parked_tid)

            resp = await client.get("/threads")
            assert resp.status_code == 200

            by_tid = {row["thread_id"]: row for row in resp.json()}
            assert parked_tid in by_tid
            assert idle_tid in by_tid
            assert by_tid[parked_tid]["has_pending_interrupt"] is True
            assert by_tid[parked_tid]["strategy_id"] == parked_sid
            assert by_tid[parked_tid]["stage"] == "paper_gate"
            assert by_tid[idle_tid]["has_pending_interrupt"] is False
            assert by_tid[idle_tid]["last_updated"] is not None
    finally:
        await _cleanup_registry_and_audits(cleanup_ids)


# ─── POST /approve — auth failures ─────────────────────────────────────


async def test_approve_without_token_returns_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No header → 401."""
    async with _app_with_paper_gate(monkeypatch) as client:
        resp = await client.post(
            "/threads/any_tid/approve",
            json={"approved": True, "notes": ""},
        )
        assert resp.status_code == 401
        assert "X-Operator-Token" in resp.json()["detail"]


async def test_approve_with_wrong_token_returns_403(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Header present but wrong value → 403 (NOT 401)."""
    async with _app_with_paper_gate(monkeypatch) as client:
        resp = await client.post(
            "/threads/any_tid/approve",
            json={"approved": True, "notes": ""},
            headers={"X-Operator-Token": "bogus"},
        )
        assert resp.status_code == 403
        assert "mismatch" in resp.json()["detail"].lower()


# ─── POST /approve — happy path ────────────────────────────────────────


async def test_approve_with_valid_token_advances_thread_and_writes_audit_and_publishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end happy path: valid token → graph advances, gate_audits
    row written, publish_gate_advanced called."""
    sid = f"sid_advance_{uuid.uuid4().hex[:8]}"
    tid = f"thread_{sid}"
    cleanup_ids = [sid]

    # Mock the publish so we don't need Redis up and can assert on the call.
    publish_mock = AsyncMock()
    monkeypatch.setattr(main_module, "publish_gate_advanced", publish_mock)

    try:
        await _seed_registry_row(sid, tid, stage="paper_gate")

        async with _app_with_paper_gate(monkeypatch) as client:
            await _park_at_paper_gate(app.state.graph, sid, tid)

            resp = await client.post(
                f"/threads/{tid}/approve",
                json={"approved": True, "notes": "ship it"},
                headers={"X-Operator-Token": TEST_OPERATOR_TOKEN},
            )

            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["resumed"] is True
            assert body["next_stage"] == "paper"
            assert isinstance(body["audit_id"], int)

            # Thread is no longer interrupted.
            post_snap = await app.state.graph.aget_state(
                {"configurable": {"thread_id": tid}}
            )
            assert not any(t.interrupts for t in post_snap.tasks)
            assert post_snap.values.get("stage") == "paper"

            # gate_audits row written with the operator_token actor.
            conn = await _open_app_conn()
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT gate, decision, actor, payload FROM gate_audits "
                        "WHERE strategy_id = %s",
                        (sid,),
                    )
                    rows = await cur.fetchall()
            finally:
                await conn.close()
            assert len(rows) == 1
            gate, decision, actor, payload = rows[0]
            assert gate == "paper"
            assert decision == "human_approve"
            assert actor.startswith("operator_token:")
            # Hash is 12 hex chars per _hash_token_for_actor.
            assert len(actor) == len("operator_token:") + 12
            # Raw token must NEVER land in the actor field.
            assert TEST_OPERATOR_TOKEN not in actor
            assert payload["decision"] == {"approved": True, "notes": "ship it"}
            assert payload["gate_node"] == "paper_gate"

            # publish_gate_advanced called with the expected payload shape.
            publish_mock.assert_awaited_once()
            ((called_tid, called_payload), _kw) = publish_mock.call_args
            assert called_tid == tid
            assert called_payload["thread_id"] == tid
            assert called_payload["decision"] == {"approved": True, "notes": "ship it"}
            assert called_payload["next_stage"] == "paper"
            assert called_payload["audit_id"] == body["audit_id"]
    finally:
        await _cleanup_registry_and_audits(cleanup_ids)


# ─── POST /approve — 409 when not interrupted ──────────────────────────


async def test_approve_when_thread_not_interrupted_returns_409(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Thread has no interrupt parked (never ran, or already advanced).
    POST /approve must refuse with 409 — silently no-op'ing a resume
    on a non-interrupted thread would mask operator mistakes."""
    sid = f"sid_terminal_{uuid.uuid4().hex[:8]}"
    tid = f"thread_{sid}"

    try:
        await _seed_registry_row(sid, tid, stage="archived")

        async with _app_with_paper_gate(monkeypatch) as client:
            # Deliberately DON'T park the thread — graph state is empty
            # for this thread_id.
            resp = await client.post(
                f"/threads/{tid}/approve",
                json={"approved": True, "notes": ""},
                headers={"X-Operator-Token": TEST_OPERATOR_TOKEN},
            )
            assert resp.status_code == 409
            assert "not parked at an interrupt" in resp.json()["detail"]
    finally:
        await _cleanup_registry_and_audits([sid])


# ─── POST /approve — concurrency / lock contract ───────────────────────


async def test_concurrent_approves_on_same_thread_serialize_via_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two concurrent approves on the same thread:
    - first acquires lock, drives past interrupt, returns 200
    - second waits for lock, then sees not-interrupted, returns 409

    This is the per-thread lock contract. Without it, both would race
    to ``astream(Command(resume=...))`` and we'd see indeterminate
    behaviour (best case: both 200 with duplicate audits; worst case:
    a partial-resume torn state)."""
    sid = f"sid_concurrent_{uuid.uuid4().hex[:8]}"
    tid = f"thread_{sid}"

    # Don't care about publish in this test — keep it cheap and avoid
    # Redis dependence.
    monkeypatch.setattr(main_module, "publish_gate_advanced", AsyncMock())

    try:
        await _seed_registry_row(sid, tid, stage="paper_gate")

        async with _app_with_paper_gate(monkeypatch) as client:
            await _park_at_paper_gate(app.state.graph, sid, tid)

            headers = {"X-Operator-Token": TEST_OPERATOR_TOKEN}
            body = {"approved": True, "notes": "concurrent"}

            r1, r2 = await asyncio.gather(
                client.post(f"/threads/{tid}/approve", json=body, headers=headers),
                client.post(f"/threads/{tid}/approve", json=body, headers=headers),
            )
            statuses = sorted([r1.status_code, r2.status_code])
            assert statuses == [200, 409], (
                f"expected exactly one 200 + one 409 (lock serialized), got {statuses}: "
                f"r1={r1.text!r} r2={r2.text!r}"
            )

            # The 409 message identifies the no-interrupt reason — not
            # an "unexpected gate node" message (which would mean the
            # graph node names were misaligned, not a lock-race).
            loser = r1 if r1.status_code == 409 else r2
            assert "not parked at an interrupt" in loser.json()["detail"]
    finally:
        await _cleanup_registry_and_audits([sid])
