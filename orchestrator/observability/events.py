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

THREAD_COMPLETED_CHANNEL: str = "ai-trading-agent:thread_completed"
"""Stage 9e. Fired when a strategy thread reaches terminal ``archived`` in the
``strategy_registry``. Consumed by ``orchestrator.supervisor_subscription``,
which debounces completions into a single coalesced supervisor run (the
event-driven half of the 9d/9e dual trigger)."""


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
    # redis.asyncio.from_url is untyped across redis-py 5.x (Stage 10a finding,
    # verified 5.0.8 + 5.2.1) — it returns a Redis at runtime. Targeted ignore
    # (not a cast): no-untyped-call (call site) and no-any-return (returned Any)
    # share the same upstream-untyped root. warn_unused_ignores flags this if a
    # future redis types from_url.
    return aioredis.from_url(url)  # type: ignore[no-untyped-call,no-any-return]


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


async def publish_kill(strategy_id: str, payload: dict[str, Any]) -> None:
    """Publish an out-of-band kill-switch fire to Redis (BRD §1.1 rule 7, §5.6).

    Channel ``ai-trading-agent:kill_switch:<strategy_id>`` (the project
    convention — multi-tenant prefix + event + id). Best-effort, same
    swallow-and-log semantics as the gate publishers: the kill switch already
    fired ``/stop`` and wrote the durable ``kill_switch_events`` row before
    calling this; a lost Redis notification is recovered when the orchestrator
    polls. The orchestrator's subscription (8g) sets
    ``artifacts.kill_switch_event`` and routes the thread to ``live_pause``.
    """
    channel = f"{KILL_SWITCH_CHANNEL}:{strategy_id}"
    await _publish(channel, payload)


async def publish_thread_completed(strategy_id: str, payload: dict[str, Any]) -> None:
    """Publish a strategy-thread completion to Redis (Stage 9e, BRD §13 row 9).

    Channel ``ai-trading-agent:thread_completed:<strategy_id>``. Mirrors
    :func:`publish_kill` exactly — self-contained (opens its own short-lived
    client via :func:`_publish`), best-effort with the same swallow-and-log
    semantics: the durable record is the ``archived`` ``strategy_registry`` row
    the caller committed BEFORE calling this, so a lost notification is
    recovered on the next supervisor cron tick (whose ``sync_registry_stage``
    sees the reconciled portfolio regardless).

    Emission contract (Stage 9e, Option 1-minimal — operator sign-off): callers
    publish AFTER the archive write commits, from exactly three REGISTRY-archive
    points — ``paper_teardown`` (paper kill / live_gate-reject), ``live_spawn``
    boot failure, and ``run_supervisor``'s post-commit retire pass. The
    graph-only archive sinks (research / validation / ``live_archive``) and
    ``sync_registry_stage`` deliberately do NOT publish; the nightly cron is the
    backstop for those funnel-internal completions. See
    ``orchestrator/supervisor_subscription.py`` for the consumer + rationale.
    """
    channel = f"{THREAD_COMPLETED_CHANNEL}:{strategy_id}"
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


async def record_kill_switch_event(
    *,
    strategy_id: str,
    reason: str,
    metrics: dict[str, Any],
    action_taken: str,
) -> int:
    """Insert a ``kill_switch_events`` row (BRD §5.8) and return its ``id``.

    Written by the out-of-band kill switch + daily-loss job (Stage 8f) on every
    fire. ``fired_at`` defaults to ``now()`` on the DB side (the migration's
    ``DEFAULT``). ``metrics`` is the profit/daily snapshot at fire time.
    Mirrors :func:`record_gate_audit`'s connection lifecycle.
    """
    sql = (
        "INSERT INTO kill_switch_events (strategy_id, reason, metrics, action_taken) "
        "VALUES (%s, %s, %s, %s) RETURNING id"
    )
    params = (strategy_id, reason, json.dumps(metrics, default=str), action_taken)

    conn = await _connect_app_db()
    try:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            row = await cur.fetchone()
        await conn.commit()
        if row is None:
            raise RuntimeError("kill_switch_events INSERT ... RETURNING id produced no row")
        return int(row[0])
    finally:
        await conn.close()


async def record_telemetry(
    conn: psycopg.AsyncConnection,
    *,
    strategy_id: str | None = None,
    stage: str | None = None,
    metrics: dict[str, Any],
    source: str,
) -> None:
    """Insert a ``telemetry`` row (BRD §5.8) on a CALLER-OWNED connection.

    Unlike :func:`record_gate_audit` / :func:`record_kill_switch_event` — each
    of which opens its OWN connection and commits — this takes the caller's
    ``conn`` and does **NOT** commit, the same convention as
    :func:`orchestrator.tools.regime.insert_regime_log`. That lets a caller
    batch a telemetry write into a larger transaction: the Stage 9 supervisor
    runner commits its registry reconciliation + spawn/retire writes + this
    decision row together, atomically.

    ``strategy_id`` is nullable — the FK exempts NULLs, and a supervisor
    decision is portfolio-level (no single strategy), so it passes ``None``.
    ``metrics`` is the JSONB payload column (NOT NULL); ``source`` tags the
    producer (e.g. ``"supervisor_decision"``). ``stage`` is optional context.
    """
    sql = "INSERT INTO telemetry (strategy_id, stage, metrics, source) VALUES (%s, %s, %s, %s)"
    params = (strategy_id, stage, json.dumps(metrics, default=str), source)
    async with conn.cursor() as cur:
        await cur.execute(sql, params)
