"""Redis thread-completion subscription — consumer side (Stage 9e, BRD §5.1, §13 row 9).

The event-driven half of the supervisor's dual trigger (the other is the 9d
nightly cron). When a strategy thread reaches terminal ``archived`` IN THE
REGISTRY, the archiving site publishes
``ai-trading-agent:thread_completed:<strategy_id>``
(:func:`orchestrator.observability.events.publish_thread_completed`). This
module is the consumer: a long-running FastAPI-lifespan task PSUBSCRIBEs the
pattern and, on each completion, schedules a single COALESCED supervisor run via
an APScheduler one-shot job (:func:`make_schedule_supervisor_run`). N completions
in quick succession debounce to ONE run that surveys the fully-settled portfolio.

Emission contract (Stage 9e, Option 1-minimal — operator sign-off): completions
are published from exactly three REGISTRY-archive points, each AFTER its archive
row commits:

  - ``paper_teardown`` (paper kill / live_gate-reject),
  - ``live_spawn`` boot failure,
  - ``run_supervisor`` post-commit retire pass.

The graph-only archive sinks (research / validation / ``live_archive``) and
``sync_registry_stage`` (point D) deliberately do NOT publish. Those
"funnel-internal" completions (a strategy dying in research/validation, or the
primary live exit — coordinator-fail / live_pause-reject) archive graph state
only; their registry transition is deferred to the next ``sync_registry_stage``,
and the nightly cron is the backstop that reacts to them. Trade-off accepted +
documented: those completions do not get a real-time supervisor reaction. For a
4-6 strategy/quarter portfolio a few-hours latency on reclaiming a research-
failed slot is immaterial, and coupling emission into 5 hot graph nodes (a
forever maintenance dependency) is not worth that value. Skipping ``sync`` (D)
also avoids the self-trigger loop where a sync-emitted event schedules a run
that syncs again.

Mirrors :mod:`orchestrator.kill_subscription` exactly in shape: a seam-injected
dispatch (``schedule_supervisor_run_fn``) so the loop is testable with a fake
Redis + a stub scheduler, a per-message try/except so one poison message never
kills the loop, and ``run_*`` / ``cancel_*`` lifecycle helpers for the lifespan.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger

from orchestrator.observability.events import THREAD_COMPLETED_CHANNEL
from orchestrator.scheduler import SUPERVISOR_JOBSTORE, SupervisorRunFn, _connect_app_db

logger = logging.getLogger(__name__)

# PSUBSCRIBE pattern — every strategy's completion channel. The publisher appends
# ``:<strategy_id>`` to THREAD_COMPLETED_CHANNEL, so ``:*`` matches all strategies.
THREAD_COMPLETED_PATTERN: str = f"{THREAD_COMPLETED_CHANNEL}:*"

# Fixed job id for the debounced one-shot — the SINGLE source of coalescing.
# replace_existing keys on it, so N completions collapse to one job.
SUPERVISOR_EVENT_ONESHOT_ID: str = "supervisor_event_oneshot"

# Default trailing-debounce window (seconds). Long enough to collapse a burst
# (operator retiring several stalled threads, or a cluster of paper/live
# completions) and to clear run_supervisor's own ~15-25s LLM latency; short
# enough that the operator only notices a delay if actively watching.
DEFAULT_DEBOUNCE_SECONDS: float = 30.0

# Misfire grace for the one-shot. Wider than the 5-min poll graces but far below
# the cron's 3600 — if the orchestrator is briefly busy (a prior run in flight,
# max_instances=1), the coalesced run still fires once it can rather than dropping.
SUPERVISOR_EVENT_GRACE_S: int = 600

# (strategy_id) → schedule a debounced supervisor run. Injected so the
# subscription loop is testable with neither a real scheduler nor a real Redis.
ScheduleSupervisorRunFn = Callable[[str], Awaitable[None]]


def _strategy_id_from_channel(channel: str) -> str:
    """Extract ``<strategy_id>`` from ``ai-trading-agent:thread_completed:<id>``.

    The channel prefix itself contains colons, so split on the LAST one —
    strategy_ids (registry ``strategy_id``) carry no colon. Mirrors
    :func:`orchestrator.kill_subscription._strategy_id_from_channel`.
    """
    return channel.rsplit(":", 1)[-1]


# ════════════════════════════════════════════════════════════════════════
# Debounced one-shot job + builder
# ════════════════════════════════════════════════════════════════════════


async def supervisor_event_job(*, run_supervisor_fn: SupervisorRunFn) -> None:
    """One debounced event-driven supervisor invocation (Stage 9e).

    Fired by the coalesced one-shot scheduled in :func:`make_schedule_supervisor_run`.
    Mirrors :func:`orchestrator.scheduler.supervisor_cron_job` exactly — opens a
    fresh app-DB connection per fire, runs the supervisor at ``trigger="event"``
    via the lifespan-bound runner (graph + store already bound into
    ``run_supervisor_fn``), logs the decision summary, and closes the connection.
    Any exception is logged and SWALLOWED: one bad event run must not crash the
    scheduler. The runner owns its own transaction (commit/rollback); this job
    only manages the connection lifecycle.
    """
    conn = await _connect_app_db()
    try:
        decision = await run_supervisor_fn(conn, trigger="event")
        actions = getattr(decision, "actions", []) or []
        rationale = str(getattr(decision, "overall_rationale", ""))
        logger.info(
            "supervisor event run: %d action(s); rationale=%s",
            len(actions),
            rationale[:200],
        )
    except Exception as exc:  # noqa: BLE001 — one bad run must not kill the scheduler
        logger.error("supervisor event run failed: %s; swallowing (subscription continues)", exc)
    finally:
        await conn.close()


def make_schedule_supervisor_run(
    scheduler: AsyncIOScheduler,
    run_supervisor_fn: SupervisorRunFn,
    *,
    debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
) -> ScheduleSupervisorRunFn:
    """Build the debounced ``schedule_supervisor_run_fn`` the subscription calls.

    Returns a coroutine that, given a ``strategy_id`` from a completion event,
    (re)schedules a SINGLE APScheduler one-shot job (fixed id
    :data:`SUPERVISOR_EVENT_ONESHOT_ID`) to fire ``debounce_seconds`` from now via
    a :class:`~apscheduler.triggers.date.DateTrigger`, with ``replace_existing=True``
    + ``max_instances=1`` in the dedicated ``supervisor_memory`` jobstore. That
    jobstore is used (not the persisted SQLAlchemy store) because the job closes
    over the non-picklable ``run_supervisor_fn`` — the SAME store the 9d cron
    uses; the FastAPI lifespan registers the cron (which creates the store)
    before wiring this subscription.

    **Debounce semantics: last-event-wins (trailing edge).** ``replace_existing=True``
    + a ``DateTrigger`` recompute ``next_run_time`` from the NEW ``run_date`` on
    each call, so each completion pushes the fire time forward. Events at T, T+5,
    T+10 with a 30s window fire ONCE at **T+40** — the supervisor reasons over the
    FULLY SETTLED burst, not a mid-burst snapshot. If a completion arrives after
    the one-shot already fired (a ``DateTrigger`` job auto-removes post-fire),
    there is no job to replace and a fresh one is scheduled — a new burst → a new
    run, exactly right. ``max_instances=1`` keeps an event run from overlapping a
    prior event run.

    The ``strategy_id`` is logged for traceability but does NOT scope the run:
    ``run_supervisor`` always surveys the whole portfolio (``sync_registry_stage``
    + snapshot) at fire time, so a coalesced run correctly reflects EVERY
    completion in the window, not just the one that happened to (re)arm it.
    """

    async def schedule_supervisor_run_fn(strategy_id: str) -> None:
        run_date = datetime.now(UTC) + timedelta(seconds=debounce_seconds)
        scheduler.add_job(
            supervisor_event_job,
            trigger=DateTrigger(run_date=run_date),
            kwargs={"run_supervisor_fn": run_supervisor_fn},
            id=SUPERVISOR_EVENT_ONESHOT_ID,
            jobstore=SUPERVISOR_JOBSTORE,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=SUPERVISOR_EVENT_GRACE_S,
        )
        logger.info(
            "scheduled debounced supervisor event run (re-armed by strategy_id=%s) "
            "to fire ~%.0fs from now (last-event-wins)",
            strategy_id,
            debounce_seconds,
        )

    return schedule_supervisor_run_fn


# ════════════════════════════════════════════════════════════════════════
# Subscription loop (mirrors kill_subscription.run_kill_subscription)
# ════════════════════════════════════════════════════════════════════════


async def _handle_message(
    message: dict[str, Any], schedule_supervisor_run_fn: ScheduleSupervisorRunFn
) -> None:
    """Parse one pubsub message and dispatch to the debounced scheduler.

    Malformed messages (bad JSON, non-object payload, unparseable channel) are
    logged and skipped — a poisoned message must not kill the subscription. A
    scheduler failure (e.g. the jobstore is momentarily unavailable) is also
    caught so one bad event does not abort the loop. The payload is parsed for
    malformed-safety, but the dispatch needs only the ``strategy_id``: the
    coalesced run surveys the whole portfolio regardless.
    """
    channel = message.get("channel")
    if isinstance(channel, bytes):
        channel = channel.decode("utf-8")

    try:
        data = message.get("data")
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        if not isinstance(data, str):
            raise ValueError(f"thread_completed payload data is not text: {type(data).__name__}")
        event = json.loads(data)
        if not isinstance(event, dict):
            raise ValueError(
                f"thread_completed payload is not a JSON object: {type(event).__name__}"
            )
        strategy_id = _strategy_id_from_channel(str(channel))
        if not strategy_id:
            raise ValueError(f"could not parse strategy_id from channel {channel!r}")
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning(
            "supervisor subscription: skipping malformed message channel=%r err=%s",
            channel,
            exc,
        )
        return

    try:
        await schedule_supervisor_run_fn(strategy_id)
    except Exception as exc:  # noqa: BLE001 — a scheduler failure must not kill the loop
        logger.error(
            "supervisor subscription: schedule failed sid=%s err=%s; continuing",
            strategy_id,
            exc,
        )


async def run_supervisor_subscription(
    redis_client: Any,
    *,
    schedule_supervisor_run_fn: ScheduleSupervisorRunFn,
    pattern: str = THREAD_COMPLETED_PATTERN,
) -> None:
    """Long-running PSUBSCRIBE loop over the thread-completed channel pattern.

    Runs as a lifespan ``asyncio.Task``; cancelled on shutdown. Cleans up the
    subscription (punsubscribe + aclose) on any exit — cancel or natural end.
    Mirrors :func:`orchestrator.kill_subscription.run_kill_subscription`.

    Parameters
    ----------
    redis_client
        A ``redis.asyncio.Redis`` (or test double) exposing ``pubsub()``.
    schedule_supervisor_run_fn
        Where each completion is dispatched — the debounced one-shot scheduler
        from :func:`make_schedule_supervisor_run`.
    pattern
        PSUBSCRIBE glob (default :data:`THREAD_COMPLETED_PATTERN`).
    """
    pubsub = redis_client.pubsub()
    try:
        await pubsub.psubscribe(pattern)
        logger.info("supervisor subscription active pattern=%s", pattern)
        async for message in pubsub.listen():
            if message.get("type") not in ("pmessage", "message"):
                # subscribe-confirmation frames carry no payload.
                continue
            await _handle_message(message, schedule_supervisor_run_fn)
    except asyncio.CancelledError:
        logger.info("supervisor subscription cancelled; cleaning up")
        raise
    finally:
        try:
            await pubsub.punsubscribe(pattern)
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass
        try:
            await pubsub.aclose()
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass


async def cancel_supervisor_subscription(task: asyncio.Task[None]) -> None:
    """Cancel the subscription task and await its cleanup (lifespan shutdown)."""
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
