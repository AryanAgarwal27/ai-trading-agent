"""Stage 6h integration tests — full paper_gate resume cycle via FastAPI.

End-to-end automated equivalent of the Stage 6f mid-stage manual smoke:

1. Park a thread at ``paper_gate`` using the production smoke-graph
   helper (``_build_paper_gate_only_graph_for_smoke``).
2. Hit ``POST /threads/{tid}/approve`` with the operator token via
   ``httpx.AsyncClient``.
3. Verify the response shape, the ``gate_audits`` row (BOTH the
   top-level ``payload.notes`` and the nested
   ``payload.decision.notes`` paths per the 6f fix), and the Redis
   publish call.

Key difference from
``tests/integration/test_main_endpoints.py``: that file uses a local
``_build_paper_gate_test_graph`` helper with an inline node — this
file uses the SAME production helper the smoke override invokes when
``AIT_SMOKE_PAPER_GATE_GRAPH=1``, so the production smoke code path
is automated here. Both files cover complementary surfaces.

All ``@pytest.mark.integration``. Requires Postgres reachable at the
env URIs. Redis is mocked at the ``events._redis_client`` seam so the
test asserts on the channel-name + payload without a running Redis.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock

import psycopg
import pytest
from dotenv import load_dotenv
from httpx import ASGITransport, AsyncClient

from orchestrator.main import _build_paper_gate_only_graph_for_smoke, app
from orchestrator.observability import events

load_dotenv()

pytestmark = pytest.mark.integration

TEST_OPERATOR_TOKEN = "test-operator-token-stage6h-deadbeef"


# ─── Helpers ───────────────────────────────────────────────────────────


def _seeded_state(strategy_id: str) -> dict[str, Any]:
    """State shape the production ``paper_gate`` node reads from.

    Mirrors what risk_analyst would have written into ``gate_decisions``
    in a real research → validation flow. Same shape used by
    ``tests/integration/test_paper_gate_e2e.py`` so the resume cycle's
    interrupt payload matches the SPEC §4.1 dashboard contract.
    """
    return {
        "strategy_id": strategy_id,
        "gate_decisions": {
            "backtest": {
                "passed": True,
                "best_param_set_id": "ps_1",
                "sharpe_is": 1.7,
                "oos_ratio": 0.72,
                "max_dd": 0.18,
            },
            "robustness": {
                "passed": True,
                "monte_carlo": {"pct_5_final_equity": 1.06},
                "regime": {"regimes_passed": 3},
                "fee_stress": {"degradation_2x": 0.26, "degradation_3x": 0.41},
            },
            "risk_analyst": {
                "decision": "approve",
                "primary_concern": "fee-stress 3x near cap",
                "rationale": (
                    "Walk-forward Sharpe holds at 1.7 with 0.72 OOS ratio; "
                    "all three regime buckets pass; MC 5th-pct at 1.06."
                ),
                "confidence": 0.74,
            },
        },
    }


async def _open_app_conn() -> psycopg.AsyncConnection:
    return await psycopg.AsyncConnection.connect(events._libpq_dsn(os.environ["DATABASE_URL"]))


async def _seed_registry_row(strategy_id: str, thread_id: str) -> None:
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
                    "paper_gate",
                    json.dumps(["BTC/USDT"]),
                    "5m",
                ),
            )
        await conn.commit()
    finally:
        await conn.close()


async def _cleanup(strategy_ids: list[str]) -> None:
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


@asynccontextmanager
async def _app_with_smoke_graph(
    monkeypatch: pytest.MonkeyPatch,
    mock_redis: AsyncMock,
) -> AsyncIterator[AsyncClient]:
    """Drive lifespan, swap in the PRODUCTION smoke graph, mock Redis.

    The swap uses the same helper the FastAPI lifespan invokes when
    ``AIT_SMOKE_PAPER_GATE_GRAPH=1`` — so this test exercises the
    actual production smoke code path, not a bespoke fixture.
    """
    monkeypatch.setenv("OPERATOR_TOKEN", TEST_OPERATOR_TOKEN)
    # Mock at the _redis_client seam so publish_gate_advanced's call
    # path runs in full (channel-name formatting, JSON serialization,
    # finally-aclose) — only the actual socket I/O is replaced.
    monkeypatch.setattr(events, "_redis_client", lambda: mock_redis)
    async with app.router.lifespan_context(app):
        app.state.graph = _build_paper_gate_only_graph_for_smoke(app.state.saver)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def _park_at_paper_gate(strategy_id: str, thread_id: str) -> None:
    """Drive the (already-installed) smoke graph until it parks at paper_gate."""
    config = {"configurable": {"thread_id": thread_id}}
    async for _ in app.state.graph.astream(_seeded_state(strategy_id), config=config):
        pass


def _filter_advanced(publish_calls: list) -> list:
    """gate_pending fires during the initial park; gate_advanced fires
    on resume. Filter to the resume-side publishes only."""
    return [
        c for c in publish_calls
        if c.args[0].startswith("ai-trading-agent:gate_advanced:")
    ]


# ─── 1. Happy path ─────────────────────────────────────────────────────


async def test_paper_gate_full_resume_cycle_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Approve → response 200, audit row landed at BOTH notes paths,
    Redis publish fired with the correct channel + payload."""
    sid = f"sid_6h_happy_{uuid.uuid4().hex[:8]}"
    tid = f"thread_{sid}"
    notes_text = "looks ready for paper"
    mock_redis = AsyncMock()

    try:
        await _seed_registry_row(sid, tid)
        async with _app_with_smoke_graph(monkeypatch, mock_redis) as client:
            await _park_at_paper_gate(sid, tid)

            resp = await client.post(
                f"/threads/{tid}/approve",
                json={"approved": True, "notes": notes_text},
                headers={"X-Operator-Token": TEST_OPERATOR_TOKEN},
            )

            # ── Response shape.
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["resumed"] is True
            assert body["next_stage"] == "paper"
            assert isinstance(body["audit_id"], int)

            # ── Graph state advanced past the interrupt.
            post_snap = await app.state.graph.aget_state(
                {"configurable": {"thread_id": tid}}
            )
            assert not any(t.interrupts for t in post_snap.tasks)
            assert post_snap.values.get("stage") == "paper"
            paper_block = post_snap.values["gate_decisions"]["paper"]
            assert paper_block["approved"] is True
            assert paper_block["notes"] == notes_text
            assert paper_block["by"] == "human"

            # ── gate_audits row.
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
            assert len(rows) == 1, f"expected exactly 1 gate_audits row, got {len(rows)}"
            gate, decision, actor, payload = rows[0]
            assert gate == "paper"
            assert decision == "human_approve"
            assert actor.startswith("operator_token:")
            assert TEST_OPERATOR_TOKEN not in actor, "raw token must NEVER appear in actor"
            # BOTH notes paths (per the 6f fix — top-level convenience
            # AND nested canonical) must resolve to the same value.
            assert payload["notes"] == notes_text, (
                f"top-level payload.notes mismatch: {payload!r}"
            )
            assert payload["decision"]["notes"] == notes_text, (
                f"nested payload.decision.notes mismatch: {payload!r}"
            )
            assert payload["decision"]["approved"] is True
            assert payload["gate_node"] == "paper_gate"

            # ── Redis publish (gate_advanced channel only — gate_pending
            # also fires during park, filter to the resume side).
            advanced = _filter_advanced(mock_redis.publish.call_args_list)
            assert len(advanced) == 1, (
                f"expected exactly 1 gate_advanced publish; got {len(advanced)}: "
                f"all calls = {mock_redis.publish.call_args_list!r}"
            )
            channel, body_bytes = advanced[0].args
            assert channel == f"ai-trading-agent:gate_advanced:{tid}"
            published = json.loads(body_bytes)
            assert published["thread_id"] == tid
            assert published["next_stage"] == "paper"
            assert published["audit_id"] == body["audit_id"]
            assert published["decision"] == {"approved": True, "notes": notes_text}
    finally:
        await _cleanup([sid])


# ─── 2. Reject path ────────────────────────────────────────────────────


async def test_paper_gate_full_resume_cycle_reject_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject → response 200, next_stage='archived', audit row with
    decision='human_reject', graph state's failure_reason carries the
    operator's notes."""
    sid = f"sid_6h_reject_{uuid.uuid4().hex[:8]}"
    tid = f"thread_{sid}"
    notes_text = "regime drift concerns"
    mock_redis = AsyncMock()

    try:
        await _seed_registry_row(sid, tid)
        async with _app_with_smoke_graph(monkeypatch, mock_redis) as client:
            await _park_at_paper_gate(sid, tid)

            resp = await client.post(
                f"/threads/{tid}/approve",
                json={"approved": False, "notes": notes_text},
                headers={"X-Operator-Token": TEST_OPERATOR_TOKEN},
            )

            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["resumed"] is True
            assert body["next_stage"] == "archived"
            assert isinstance(body["audit_id"], int)

            # ── Graph state archived with the operator's notes in
            # failure_reason (paper_gate's reject path encodes this).
            post_snap = await app.state.graph.aget_state(
                {"configurable": {"thread_id": tid}}
            )
            assert post_snap.values.get("stage") == "archived"
            failure = post_snap.values.get("failure_reason", "")
            assert "paper_gate_rejected" in failure, (
                f"failure_reason must carry the canonical prefix; got {failure!r}"
            )
            assert notes_text in failure, (
                f"failure_reason must include operator notes; got {failure!r}"
            )

            # ── gate_audits row reflects the reject.
            conn = await _open_app_conn()
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT decision, payload FROM gate_audits "
                        "WHERE strategy_id = %s",
                        (sid,),
                    )
                    rows = await cur.fetchall()
            finally:
                await conn.close()
            assert len(rows) == 1
            decision, payload = rows[0]
            assert decision == "human_reject"
            assert payload["notes"] == notes_text
            assert payload["decision"]["approved"] is False
            assert payload["decision"]["notes"] == notes_text
    finally:
        await _cleanup([sid])


# ─── 3. Auth sanity ────────────────────────────────────────────────────


async def test_paper_gate_resume_without_token_returns_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-asserted here for documentation purposes: the auth contract
    holds on the production-smoke-graph code path too, not just on the
    bespoke test graph in test_main_endpoints.py."""
    sid = f"sid_6h_noauth_{uuid.uuid4().hex[:8]}"
    tid = f"thread_{sid}"
    mock_redis = AsyncMock()

    try:
        await _seed_registry_row(sid, tid)
        async with _app_with_smoke_graph(monkeypatch, mock_redis) as client:
            await _park_at_paper_gate(sid, tid)

            resp = await client.post(
                f"/threads/{tid}/approve",
                json={"approved": True, "notes": ""},
                # No X-Operator-Token header.
            )
            assert resp.status_code == 401
            assert "X-Operator-Token" in resp.json()["detail"]

            # ── Side-effect check: no audit row, no gate_advanced
            # publish, graph still parked at the interrupt.
            conn = await _open_app_conn()
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT COUNT(*) FROM gate_audits WHERE strategy_id = %s",
                        (sid,),
                    )
                    count_row = await cur.fetchone()
            finally:
                await conn.close()
            assert count_row is not None and count_row[0] == 0, (
                "401 path must NOT write a gate_audits row"
            )

            assert not _filter_advanced(mock_redis.publish.call_args_list), (
                "401 path must NOT fire a gate_advanced publish"
            )

            post_snap = await app.state.graph.aget_state(
                {"configurable": {"thread_id": tid}}
            )
            assert any(
                t.interrupts for t in post_snap.tasks
            ), "thread must remain parked after a 401 — endpoint must not advance state"
    finally:
        await _cleanup([sid])
