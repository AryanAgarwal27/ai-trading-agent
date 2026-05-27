"""Unit + integration tests for :mod:`orchestrator.observability.events` (Stage 6c).

Three mocked-unit tests cover the channel-formatting / error-swallow /
SQL-shape contracts without needing Redis or Postgres.

The fourth test (``test_audit_decision_constraint_rejects_invalid_decision``)
is the only one that can NOT be mocked — a CHECK constraint is a DB
feature, not a Python check. It's marked ``@pytest.mark.integration``
and requires a live Postgres on ``DATABASE_URL`` with the app DB
migrated (same precondition as ``test_postgres_lifecycle``).
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any
from unittest.mock import AsyncMock

import pytest
from dotenv import load_dotenv

from orchestrator.observability import events
from orchestrator.observability.events import (
    GATE_ADVANCED_CHANNEL,
    GATE_PENDING_CHANNEL,
    publish_gate_advanced,
    publish_gate_pending,
    record_gate_audit,
)

# ─── publish_gate_pending — channel-formatting + payload ───────────────


async def test_publish_gate_pending_publishes_to_correct_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Channel must be ``ai-trading-agent:gate_pending:<thread_id>`` and
    the payload must be JSON-serialized."""
    mock_client = AsyncMock()
    monkeypatch.setattr(events, "_redis_client", lambda: mock_client)

    payload = {"kind": "paper_gate", "strategy_id": "abc", "summary": {"x": 1}}
    await publish_gate_pending("strategy_abc", payload)

    mock_client.publish.assert_awaited_once()
    args, _ = mock_client.publish.call_args
    channel, body = args
    assert channel == f"{GATE_PENDING_CHANNEL}:strategy_abc"
    assert channel == "ai-trading-agent:gate_pending:strategy_abc"
    # Payload arrives JSON-serialized. Round-tripping confirms shape.
    assert json.loads(body) == payload
    # Best-effort client lifecycle — aclose must be awaited.
    mock_client.aclose.assert_awaited_once()


async def test_publish_gate_advanced_uses_advanced_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Belt-and-braces: confirms the advanced publisher targets a
    DIFFERENT channel from gate_pending. Mixing them up would silently
    misroute dashboard updates."""
    mock_client = AsyncMock()
    monkeypatch.setattr(events, "_redis_client", lambda: mock_client)

    await publish_gate_advanced("strategy_xyz", {"resumed": True})

    channel, _body = mock_client.publish.call_args.args
    assert channel == f"{GATE_ADVANCED_CHANNEL}:strategy_xyz"
    assert channel == "ai-trading-agent:gate_advanced:strategy_xyz"
    assert channel != f"{GATE_PENDING_CHANNEL}:strategy_xyz"


# ─── publish_*  — error-swallow semantics ──────────────────────────────


async def test_publish_swallows_redis_connection_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Redis being down must NOT propagate — graph threads inside
    ``interrupt()`` cannot tolerate a publisher raising."""
    mock_client = AsyncMock()
    mock_client.publish.side_effect = ConnectionError("connection refused")
    monkeypatch.setattr(events, "_redis_client", lambda: mock_client)

    # If this raised, pytest would fail the test — the assertion is the
    # absence of an exception.
    await publish_gate_pending("strategy_dead_redis", {"kind": "paper_gate"})

    # Verify the failure WAS logged (silent swallow would be worse than
    # crashing — operator needs to see it in the FastAPI logs).
    assert any(
        "redis publish failed" in rec.getMessage() for rec in caplog.records
    ), "expected a 'redis publish failed' warning in the log stream"
    # aclose still attempted even on publish failure.
    mock_client.aclose.assert_awaited_once()


# ─── record_gate_audit — SQL shape (mocked psycopg) ────────────────────


async def test_record_gate_audit_writes_row_with_correct_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the INSERT SQL, parameter order, JSON serialization, and
    the returned row id — without touching real Postgres."""

    # Build a layered AsyncMock that satisfies:
    #   conn = await _connect_app_db()
    #   async with conn.cursor() as cur:
    #       await cur.execute(sql, params)
    #       row = await cur.fetchone()
    #   await conn.commit()
    #   await conn.close()
    captured: dict[str, Any] = {}

    cur = AsyncMock()
    cur.fetchone.return_value = (4242,)

    async def _execute(sql: str, params: tuple[Any, ...]) -> None:
        captured["sql"] = sql
        captured["params"] = params

    cur.execute.side_effect = _execute

    # async-with on cursor() returns the cursor itself.
    cursor_cm = AsyncMock()
    cursor_cm.__aenter__.return_value = cur
    cursor_cm.__aexit__.return_value = None

    conn = AsyncMock()
    conn.cursor = lambda: cursor_cm

    async def _fake_connect() -> AsyncMock:
        return conn

    monkeypatch.setattr(events, "_connect_app_db", _fake_connect)

    audit_id = await record_gate_audit(
        strategy_id="strategy_unit_001",
        gate="paper",
        decision="human_approve",
        actor="operator:aryan",
        payload={"notes": "looks good", "by_ui": "streamlit"},
    )

    # Returned id matches the mocked RETURNING value.
    assert audit_id == 4242

    # SQL targets gate_audits with the right column ordering + RETURNING.
    assert "INSERT INTO gate_audits" in captured["sql"]
    assert (
        "(strategy_id, gate, decision, actor, payload)" in captured["sql"]
    ), "column ordering must match the (strategy_id, gate, decision, actor, payload) contract"
    assert "RETURNING id" in captured["sql"]

    # Params: positional order matches the column ordering. Payload is
    # JSON-serialized so psycopg can cast into JSONB.
    params = captured["params"]
    assert params[0] == "strategy_unit_001"
    assert params[1] == "paper"
    assert params[2] == "human_approve"
    assert params[3] == "operator:aryan"
    assert json.loads(params[4]) == {"notes": "looks good", "by_ui": "streamlit"}

    # Lifecycle: commit + close both awaited (durability + no conn leak).
    conn.commit.assert_awaited_once()
    conn.close.assert_awaited_once()


# ─── record_gate_audit — real DB CHECK constraint ──────────────────────


@pytest.mark.integration
async def test_audit_decision_constraint_rejects_invalid_decision() -> None:
    """The ``gate_audits.decision`` column has a CHECK constraint per BRD
    §5.8. An invalid decision must raise — not silently land in the
    table. Requires a running Postgres at ``DATABASE_URL`` with the
    app DB migrated (same precondition as ``test_postgres_lifecycle``)."""
    import psycopg

    load_dotenv()  # ensures DATABASE_URL is populated if the operator runs from cwd

    if "DATABASE_URL" not in os.environ:
        pytest.skip("DATABASE_URL not set in environment")

    # Insert a registry row first so the FK on gate_audits.strategy_id holds.
    # The CHECK on decision fires regardless, but a clean FK lets us point
    # at the actual failure rather than mis-blaming the FK.
    fixture_strategy_id = f"test_check_{uuid.uuid4().hex[:8]}"
    libpq_dsn = events._libpq_dsn(os.environ["DATABASE_URL"])
    conn = await psycopg.AsyncConnection.connect(libpq_dsn)
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO strategy_registry "
                "(strategy_id, thread_id, name, template, stage, pairs, timeframe) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    fixture_strategy_id,
                    f"thread_{fixture_strategy_id}",
                    "test-check-fixture",
                    "mean_reversion_template",
                    "research",
                    json.dumps(["BTC/USDT"]),
                    "5m",
                ),
            )
        await conn.commit()

        # Now the actual assertion: invalid decision must raise. psycopg
        # surfaces CHECK violations as ``psycopg.errors.CheckViolation``.
        with pytest.raises(psycopg.errors.CheckViolation):
            await record_gate_audit(
                strategy_id=fixture_strategy_id,
                gate="paper",
                decision="weird",  # NOT in the allowed set
                actor="test",
                payload={},
            )
    finally:
        # Cleanup — the failed insert won't have committed, but the
        # strategy_registry fixture row will linger if we don't.
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM strategy_registry WHERE strategy_id = %s",
                    (fixture_strategy_id,),
                )
            await conn.commit()
        finally:
            await conn.close()
