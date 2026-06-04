"""Redis kill-switch subscription — consumer side (Stage 8g, BRD §5.6, §1.1 rule 7).

The out-of-band kill switch (``scheduler.kill_switch_poll_job``) POSTs
``/api/v1/stop`` directly and publishes
``ai-trading-agent:kill_switch:<strategy_id>`` (``events.publish_kill``). This
module is the consumer: a long-running FastAPI-lifespan task PSUBSCRIBEs the
pattern, parses each event, and writes ``artifacts.kill_switch_event`` into the
LangGraph state for the matching thread, then DIRECTLY resumes the thread (9f,
D-6 kill path). ``live._route_after_live_wait`` reads the event and routes
straight to ``live_pause`` (BRD §5.6 — no coordinator vote), and
``build_interrupt_payload`` takes the kill-switch branch.

9f (D-6 kill path closure): the resume is a direct ``Command(resume=...)``, NOT
the periodic /wake job, because the ``aupdate_state`` that writes the kill event
clears the PARENT-visible interrupt surface (the 8h finding) — so /wake would
409 the kill-written thread. Command(resume=...) straight to the thread bypasses
the endpoint's interrupt-presence guard and resumes the still-parked nested
``live_wait``. The periodic /wake path (kind="live_wait") is the SEPARATE no-kill
re-eval trigger and must never operate on a kill-written thread (regression-
guarded in tests/unit/test_live_wake.py).

Decoupling (Stage 8g design note "subscription writes to state without coupling
concerns"): the state write is injected as a seam (``kill_event_writer_fn``) so
the subscription LOOP is testable with a fake Redis client + a stub writer (no
checkpointer), and ``make_kill_event_writer`` — the ONLY code that touches the
graph — is what the lifespan wires in.

Fork-2 (kill event for a thread the graph doesn't know about): the default
writer reads ``aget_state`` first and SKIPS the state write (logging) when the
thread has no checkpoint or is not in the ``live`` stage, rather than letting
``aupdate_state`` mint a phantom checkpoint carrying only the kill event. The
durable record is the ``kill_switch_events`` row the kill switch already wrote
(8f); the state write is purely the graph-routing trigger.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.types import Command

from orchestrator.observability.events import KILL_SWITCH_CHANNEL

logger = logging.getLogger(__name__)

# Hold references to fire-and-forget kill direct-resume tasks so they aren't GC'd
# mid-run (mirrors supervisor._BACKGROUND_SPAWN_TASKS). 9f closes D-6's kill path.
_KILL_RESUME_TASKS: set[asyncio.Task[None]] = set()

# PSUBSCRIBE pattern — every strategy's kill channel. KILL_SWITCH_CHANNEL is the
# static ``ai-trading-agent:kill_switch`` prefix; the publisher appends
# ``:<strategy_id>`` (events.publish_kill), so ``:*`` matches all strategies.
KILL_SWITCH_PATTERN: str = f"{KILL_SWITCH_CHANNEL}:*"

# (strategy_id, normalized_event) → write into the graph state. Injected so
# tests need neither a real checkpointer nor a real Redis.
KillEventWriterFn = Callable[[str, dict[str, Any]], Awaitable[None]]


def _strategy_id_from_channel(channel: str) -> str:
    """Extract ``<strategy_id>`` from ``ai-trading-agent:kill_switch:<id>``.

    The channel prefix itself contains colons, so split on the LAST one —
    strategy_ids (registry ``strategy_id``) carry no colon.
    """
    return channel.rsplit(":", 1)[-1]


def _normalize_kill_event(raw: dict[str, Any]) -> dict[str, Any]:
    """Shape the published event for the dashboard kill-switch card.

    ``events.publish_kill`` (8f) sends ``{reason, fired_at, action_taken,
    metrics_summary}``; the dashboard kill-switch card
    (``build_interrupt_payload`` → ``kill_switch_card.render_kill_switch_card``)
    reads ``event["metrics"]`` and ``event["action_taken"]``. We ALIAS
    ``metrics_summary`` → ``metrics`` (rather than rename, so nothing the
    publisher sent is lost). ``action_taken`` now flows through the publish
    payload (8g Change 1 source-fix), so we do NOT inject a default — a legacy
    or malformed event without it renders as ``_unknown_`` via the card's
    fallback rather than a fabricated stop label.
    """
    event = dict(raw)
    if "metrics" not in event and "metrics_summary" in event:
        event["metrics"] = event["metrics_summary"]
    return event


def make_kill_event_writer(graph: Any) -> KillEventWriterFn:
    """Build the default ``kill_event_writer_fn`` bound to a compiled graph.

    The returned coroutine writes ``artifacts.kill_switch_event`` (merging into
    the existing artifacts dict — the channel has no reducer, so a blind write
    would drop ``live_started_at`` etc.) for ``thread_id = strategy_<id>``.

    Fork-2 guard: skip (with a log) when the thread has no checkpoint or is not
    in the ``live`` stage — never mint a phantom checkpoint for an unknown or
    already-terminal thread.
    """

    async def _writer(strategy_id: str, event: dict[str, Any]) -> None:
        thread_id = f"strategy_{strategy_id}"
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await graph.aget_state(config)
        values = snapshot.values or {}

        if not values:
            logger.warning(
                "kill event for unknown thread %s (no checkpoint); skipping "
                "state write (durable row already in kill_switch_events)",
                thread_id,
            )
            return
        if values.get("stage") != "live":
            logger.warning(
                "kill event for thread %s not in 'live' stage (stage=%s); " "skipping state write",
                thread_id,
                values.get("stage"),
            )
            return

        artifacts = dict(values.get("artifacts") or {})
        artifacts["kill_switch_event"] = _normalize_kill_event(event)
        await graph.aupdate_state(config, {"artifacts": artifacts})
        logger.warning(
            "KILL EVENT written to graph state: thread=%s reason=%s — direct-resuming",
            thread_id,
            event.get("reason"),
        )

        # 9f (D-6 kill path, Option (b) — mandated by the 8h finding): the
        # parent-level aupdate_state above CLEARED the parent-visible interrupt
        # surface (tasks[*].interrupts is now empty), so /wake would 409. But the
        # NESTED live_wait interrupt is still parked-and-resumable, so resume the
        # thread DIRECTLY here via Command(resume=...). _route_after_live_wait
        # (8g) reads artifacts.kill_switch_event and routes straight to live_pause,
        # which emits a live_pause_review interrupt — the new HITL park point the
        # operator advances via /approve. Fire-and-forget: do NOT await to
        # completion (the resumed graph runs the whole live_pause path, and the
        # subscription loop must keep consuming other events). A resume failure is
        # logged loudly; the durable kill_switch_events row (8f) already exists, so
        # the failure is recoverable manually — one crashed resume must not kill
        # the loop or strand other threads.
        async def _direct_resume() -> None:
            try:
                async for _ in graph.astream(
                    Command(resume={"wake": True, "source": "kill_subscription"}),
                    config=config,
                ):
                    pass
                logger.info("kill direct-resume complete thread=%s (now at live_pause)", thread_id)
            except Exception as exc:  # noqa: BLE001 — fire-and-forget; recoverable via the kse row
                logger.error(
                    "kill direct-resume FAILED thread=%s exc=%s — kill_switch_events row "
                    "persists; resume manually via /approve once state is sound",
                    thread_id,
                    exc,
                )

        task = asyncio.create_task(_direct_resume())
        _KILL_RESUME_TASKS.add(task)
        task.add_done_callback(_KILL_RESUME_TASKS.discard)

    return _writer


async def _handle_message(message: dict[str, Any], writer: KillEventWriterFn) -> None:
    """Parse one pubsub message and dispatch to the writer.

    Malformed messages (bad JSON, non-object payload, unparseable channel) are
    logged and skipped — a poisoned message must not kill the subscription. A
    writer failure (DB/graph error) is also caught so one bad event does not
    abort the loop.
    """
    channel = message.get("channel")
    if isinstance(channel, bytes):
        channel = channel.decode("utf-8")

    try:
        data = message.get("data")
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        if not isinstance(data, str):
            raise ValueError(f"kill payload data is not text: {type(data).__name__}")
        event = json.loads(data)
        if not isinstance(event, dict):
            raise ValueError(f"kill payload is not a JSON object: {type(event).__name__}")
        strategy_id = _strategy_id_from_channel(str(channel))
        if not strategy_id:
            raise ValueError(f"could not parse strategy_id from channel {channel!r}")
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning(
            "kill subscription: skipping malformed message channel=%r err=%s",
            channel,
            exc,
        )
        return

    event["channel"] = channel
    try:
        await writer(strategy_id, event)
    except Exception as exc:  # noqa: BLE001 — a writer failure must not kill the loop
        logger.error(
            "kill subscription: writer failed sid=%s err=%s; continuing",
            strategy_id,
            exc,
        )


async def run_kill_subscription(
    redis_client: Any,
    *,
    kill_event_writer_fn: KillEventWriterFn,
    pattern: str = KILL_SWITCH_PATTERN,
) -> None:
    """Long-running PSUBSCRIBE loop over the kill-switch channel pattern.

    Runs as a lifespan ``asyncio.Task``; cancelled on shutdown. Cleans up the
    subscription (punsubscribe + aclose) on any exit — cancel or natural end.

    Parameters
    ----------
    redis_client
        An ``redis.asyncio.Redis`` (or test double) exposing ``pubsub()``.
    kill_event_writer_fn
        Where parsed events go (default from :func:`make_kill_event_writer`).
    pattern
        PSUBSCRIBE glob (default :data:`KILL_SWITCH_PATTERN`).
    """
    pubsub = redis_client.pubsub()
    try:
        await pubsub.psubscribe(pattern)
        logger.info("kill subscription active pattern=%s", pattern)
        async for message in pubsub.listen():
            if message.get("type") not in ("pmessage", "message"):
                # subscribe-confirmation frames carry no payload.
                continue
            await _handle_message(message, kill_event_writer_fn)
    except asyncio.CancelledError:
        logger.info("kill subscription cancelled; cleaning up")
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


async def cancel_kill_subscription(task: asyncio.Task[None]) -> None:
    """Cancel the subscription task and await its cleanup (lifespan shutdown)."""
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
