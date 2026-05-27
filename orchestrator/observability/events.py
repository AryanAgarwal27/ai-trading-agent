"""Pubsub publishers + ``gate_audits`` writer (Stage 6c).

Two responsibilities, kept in the same module because they share the
"event happened, fan it out to the dashboard + the audit trail" call
pattern:

1. **Redis pubsub publishers** — fire-and-forget gate-event notifications
   for the FastAPI WS layer (Stage 6d) and any future dashboard process.
   Errors are logged and swallowed: Redis being unhealthy must NOT crash
   a graph thread mid-interrupt. Lost notifications are recoverable from
   the ``gate_audits`` table on the dashboard's next poll.

2. **``gate_audits`` writer** — durable row in the ``app`` DB recording
   every gate decision (auto and human). Schema lives in
   ``db/migrations/versions/0001_init.py`` per BRD §5.8. This is the
   source of truth for "what gate decisions has this strategy seen?" —
   the pubsub stream is the realtime overlay on top.

Channel naming convention: ``ai-trading-agent:<event>:<thread_id>``.
The ``ai-trading-agent:`` prefix is a multi-tenant guard so a shared
Redis (rare for v1, but possible in dev) does not cross-talk with other
projects. The ``<thread_id>`` suffix lets a dashboard client subscribe
to a single strategy without a global firehose.

BRD §1.1 rule 7: the out-of-band kill switch publishes directly here
(``KILL_SWITCH_CHANNEL``) — its consumer is APScheduler, not the graph.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import psycopg
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


# ─── Channel name prefixes (handoff §6) ────────────────────────────────
# Format: ``ai-trading-agent:<event>:<thread_id>``. The prefix here is
# the static portion; callers append ``:<thread_id>``.

GATE_PENDING_CHANNEL: str = "ai-trading-agent:gate_pending"
"""Fired when a gate node hits ``interrupt()``. Payload: interrupt payload."""

GATE_ADVANCED_CHANNEL: str = "ai-trading-agent:gate_advanced"
"""Fired when a thread is resumed past a gate. Payload: decision + audit_id."""

KILL_SWITCH_CHANNEL: str = "ai-trading-agent:kill_switch"
"""Reserved for Stage 8. Fired by the out-of-band APScheduler kill switch."""

TELEMETRY_CHANNEL: str = "ai-trading-agent:telemetry"
"""Reserved for Stage 7+. Fired on each paper-monitor wake."""

AUDIT_CHANNEL: str = "ai-trading-agent:audit"
"""Reserved for Stage 10. Generic audit-trail tail for ops dashboards."""


# ─── DSN normalization ─────────────────────────────────────────────────


def _libpq_dsn(database_url: str) -> str:
    """Strip the SQLAlchemy ``+psycopg`` dialect suffix so raw psycopg accepts it.

    The repo's ``DATABASE_URL`` follows SQLAlchemy URL convention
    (``postgresql+psycopg://...``) because Alembic needs it. Raw psycopg
    expects a libpq DSN (``postgresql://...``). One-line normalize is
    cheaper than maintaining two env vars.
    """
    return database_url.replace("postgresql+psycopg://", "postgresql://", 1)


# ─── Connection helpers (overrideable in tests) ────────────────────────


def _redis_client() -> aioredis.Redis:
    """Open an async Redis client from ``REDIS_URL``.

    Returns a fresh client per call. Cheap (no connect-on-construct in
    redis-py 5.x — connection happens lazily on first command). Tests
    monkeypatch this to return an ``AsyncMock``.
    """
    url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
    return aioredis.from_url(url)


async def _connect_app_db() -> psycopg.AsyncConnection:
    """Open a fresh async psycopg connection to the app DB.

    The handoff calls out explicitly: do NOT reuse the LangGraph
    checkpointer's pool. Audit writes are short-lived, transactional,
    and must complete even if the checkpointer pool is exhausted.
    Tests monkeypatch this helper to return an ``AsyncMock``-like.
    """
    return await psycopg.AsyncConnection.connect(_libpq_dsn(os.environ["DATABASE_URL"]))


# ─── Publishers ────────────────────────────────────────────────────────


async def publish_gate_pending(thread_id: str, payload: dict[str, Any]) -> None:
    """Publish a "gate is parked at interrupt()" event to Redis.

    Best-effort. Redis errors are logged and swallowed — losing a
    notification is recoverable (dashboard polls ``gate_audits``), but
    crashing a graph thread inside ``interrupt()`` would strand the
    checkpoint mid-state.

    Parameters
    ----------
    thread_id
        The LangGraph thread id (``strategy_<uuid>`` for per-strategy
        threads, ``supervisor`` for the supervisor).
    payload
        The interrupt payload (from
        :func:`orchestrator.gates.hitl.build_interrupt_payload`).
    """
    channel = f"{GATE_PENDING_CHANNEL}:{thread_id}"
    await _publish(channel, payload)


async def publish_gate_advanced(thread_id: str, payload: dict[str, Any]) -> None:
    """Publish a "gate resumed, thread is moving" event to Redis.

    Symmetric to :func:`publish_gate_pending` — fires after the resume
    handler has driven the graph past the gate node and written the
    ``gate_audits`` row.
    """
    channel = f"{GATE_ADVANCED_CHANNEL}:{thread_id}"
    await _publish(channel, payload)


async def _publish(channel: str, payload: dict[str, Any]) -> None:
    """Shared publisher with the error-swallow semantics described above."""
    client = _redis_client()
    try:
        await client.publish(channel, json.dumps(payload, default=str))
    except Exception as exc:  # noqa: BLE001 — explicit fire-and-forget on any failure
        # ``default=str`` above also protects against datetime/uuid payloads
        # that would otherwise raise TypeError before reaching Redis.
        logger.warning(
            "redis publish failed channel=%s err=%s; swallowing (best-effort)",
            channel,
            exc,
        )
    finally:
        # ``aclose`` (redis-py ≥ 5.0) replaces ``close``; fire-and-forget
        # client lifecycle — we do not pool here.
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001
            pass


# ─── gate_audits writer ────────────────────────────────────────────────


async def record_gate_audit(
    *,
    strategy_id: str,
    gate: str,
    decision: str,
    actor: str,
    payload: dict[str, Any],
) -> int:
    """Insert a row into ``gate_audits`` and return its ``id``.

    Schema (BRD §5.8, migration ``0001_init``):

    - ``gate`` ∈ {``backtest``, ``paper``, ``live``, ``live_pause``}
    - ``decision`` ∈ {``auto_pass``, ``auto_fail``, ``human_approve``,
      ``human_reject``, ``human_revise``}
    - ``actor`` — free text (``"system"``, operator id, agent name)
    - ``payload`` — JSONB; we serialize via ``json.dumps`` matching the
      convention in :func:`orchestrator.tools.regime.insert_regime_log`.

    A fresh psycopg connection is opened from ``DATABASE_URL`` per call.
    Auto-commits and closes — callers don't manage the connection
    lifecycle (unlike :func:`insert_regime_log` which takes a caller-
    owned conn because the regime APScheduler job batches inserts).

    Parameters
    ----------
    strategy_id
        Must exist in ``strategy_registry`` (FK). Tests insert a fixture
        registry row first.
    gate
        One of the CHECK-constrained values above. Invalid values raise
        ``psycopg.errors.CheckViolation``.
    decision
        One of the CHECK-constrained values above. Same enforcement.
    actor
        Who/what made the decision. ``"system"`` for auto gates, an
        operator identifier for HITL gates.
    payload
        Free-form JSONB blob — at minimum the gate-relevant subset of
        state for post-hoc audit. Stage 6d's resume handler writes the
        operator's ``notes`` + decision into this column.

    Returns
    -------
    The newly inserted ``id`` (``BIGSERIAL``).
    """
    sql = (
        "INSERT INTO gate_audits (strategy_id, gate, decision, actor, payload) "
        "VALUES (%s, %s, %s, %s, %s) RETURNING id"
    )
    params = (strategy_id, gate, decision, actor, json.dumps(payload, default=str))

    conn = await _connect_app_db()
    try:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            row = await cur.fetchone()
        await conn.commit()
        if row is None:
            raise RuntimeError("gate_audits INSERT ... RETURNING id produced no row")
        return int(row[0])
    finally:
        await conn.close()
