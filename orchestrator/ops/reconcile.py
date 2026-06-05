"""On-startup disaster-recovery reconciliation (Stage 10f, BRD §16, §17 #12).

When the orchestrator restarts, the LangGraph checkpoints survive (Postgres) but
the Freqtrade containers may not — a host reboot, a crashed ``docker compose``,
an OOM-killed worker. BRD §16:

  > Reconciliation on startup: ``ops/reconcile.py`` reads ``strategy_registry``,
  > pings each ``freqtrade_api_url``; if reachable → keep state; if not →
  > transition to ``live_pause`` and write a ``kill_switch_events`` row with
  > ``reason="orchestrator_restart_no_freqtrade"``.

This module implements that contract by REUSING existing seams rather than
inventing new ones:

- **Scan** — the exporter's :func:`_active_containers` already lists the
  non-archived ``strategy_registry`` rows carrying a ``freqtrade_api_url`` (the
  running paper/live containers); reconcile reuses it verbatim.
- **Ping** — the same :class:`~orchestrator.tools.freqtrade_api.FreqtradeAPI`
  client + :func:`_default_client_factory` (5s timeout) the exporter uses; the
  unauthenticated ``/api/v1/ping`` is the lightest reachability probe (BRD §7.2).
- **Transition** — the 9f kill mechanism (:func:`make_kill_event_writer`, the
  SAME ``app.state.kill_event_writer_fn`` the Redis kill subscription drives):
  it writes ``artifacts.kill_switch_event`` onto the thread (which clears the
  PARENT-visible interrupt — the 8h finding) and then DIRECTLY resumes the
  still-parked nested ``live_wait`` via ``Command(resume=...)``.
  ``_route_after_live_wait`` reads ``artifacts.kill_switch_event`` and routes
  straight to ``live_pause`` (BRD §5.6 — no coordinator vote). This is exactly
  why D-6 closed with Option (b): ``/wake`` would 409 a kill-written thread, so
  reconcile must use the direct-resume path, NOT an endpoint. Reconcile reuses
  that path — it does not add a second one.
- **Durable row** — :func:`record_kill_switch_event` writes the
  ``kill_switch_events`` row (the recovery anchor), exactly as the out-of-band
  kill switch does on every fire (8f).

**Paper vs live (the load-bearing decision).** ``live_pause`` is a
live-subgraph-only node — a thread parked at ``paper_wait`` has no ``live_pause``
to route to, and the kill writer's Fork-2 guard SKIPS any thread whose graph
``stage != "live"``. So reconcile ACTS (row + live_pause transition) only on
``stage == "live"`` rows. A non-live container that is unreachable on restart
(paper) is LOGGED but not transitioned and writes no ``kill_switch_events`` row:
its own paper-subgraph wake re-attaches it, and its gate is HITL regardless.
Reconcile still PINGS every non-archived container (reachability is useful
operational signal) — it just gates the action on the registry stage.

**Graceful degradation (mandatory, BRD §17 #12 spirit — reconcile must not
depend on the graph being awake or on every container being up).** A single
unreachable container is the NORMAL case this handles, not an error. Each
per-container coroutine swallows its own failure so one bad row never aborts the
sweep; a DB failure listing the containers is swallowed (reconcile returns,
having done nothing); and the lifespan wraps the whole call so a reconcile error
can never crash startup.

Ordering invariant (mirrors the kill path): the durable ``kill_switch_events``
row is written BEFORE the graph transition is driven, so the recovery record
exists even if the resume fails (the resume is fire-and-forget inside the kill
writer; a failure there is logged, and ``/approve`` can advance the thread
manually off the durable row).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from orchestrator.kill_subscription import KillEventWriterFn
from orchestrator.observability.events import _connect_app_db, record_kill_switch_event
from orchestrator.observability.freqtrade_exporter import (
    ClientFactory,
    ConnectFn,
    _active_containers,
    _default_client_factory,
    _password_for_stage,
)
from orchestrator.tools.freqtrade_api import FreqtradeCredentials

logger = logging.getLogger(__name__)

# BRD §16 EXACT reason string for the kill_switch_events row.
RECONCILE_REASON = "orchestrator_restart_no_freqtrade"

# action_taken for the reconcile row. Deliberately NOT one of the scheduler's
# stop-actions (``POST /api/v1/stop`` etc.): reconcile issued NO /stop — the
# container is unreachable — so it must not trip the kill-switch idempotency
# guard (which checks kill_switch_events for a prior stop-action). It records
# that the thread was routed to live_pause for HITL review.
RECONCILE_ACTION = "routed_to_live_pause (orchestrator restart; container unreachable)"

# Injected so tests need neither a real DB row writer nor a real graph.
RecordKillFn = Callable[..., Awaitable[int]]


async def _is_reachable(url: str, stage: str, client_factory: ClientFactory) -> bool:
    """Ping one container's ``/api/v1/ping``. True iff it answers; any error
    (transport, non-200, auth) means unreachable. NEVER raises."""
    try:
        creds = FreqtradeCredentials(username="freqtrader", password=_password_for_stage(stage))
        async with client_factory(url, creds) as client:
            await client.ping()
    except Exception as exc:  # noqa: BLE001 — unreachable is the normal case, not fatal
        logger.info(
            "reconcile: container unreachable stage=%s url=%s err=%s",
            stage,
            url,
            exc,
        )
        return False
    return True


async def _reconcile_one(
    strategy_id: str,
    stage: str,
    url: str,
    *,
    kill_event_writer_fn: KillEventWriterFn,
    client_factory: ClientFactory,
    record_kill_switch_event_fn: RecordKillFn,
) -> str:
    """Reconcile one container row. Returns a bucket label for the summary:
    ``"reachable"`` | ``"transitioned"`` | ``"skipped_non_live"`` | ``"error"``.
    NEVER raises — one bad row must not abort the sweep."""
    try:
        if await _is_reachable(url, stage, client_factory):
            return "reachable"

        # Unreachable. live_pause + kill_switch_events row are live-only (see
        # module docstring: a paper thread has no live_pause node, and the kill
        # writer's Fork-2 guard would skip it anyway).
        if stage != "live":
            logger.warning(
                "reconcile: non-live container unreachable on restart "
                "(sid=%s stage=%s url=%s) — no live_pause transition; its own "
                "subgraph wake re-attaches",
                strategy_id,
                stage,
                url,
            )
            return "skipped_non_live"

        metrics = {
            "detail": "freqtrade container unreachable on orchestrator startup",
            "freqtrade_api_url": url,
        }
        # Durable row FIRST (recovery anchor), then drive the transition — so the
        # kill_switch_events record exists even if the resume later fails.
        await record_kill_switch_event_fn(
            strategy_id=strategy_id,
            reason=RECONCILE_REASON,
            metrics=metrics,
            action_taken=RECONCILE_ACTION,
        )
        logger.warning(
            "reconcile: LIVE container unreachable on restart (sid=%s url=%s) — "
            "wrote kill_switch_events row + routing to live_pause",
            strategy_id,
            url,
        )
        # Reuse the 9f kill writer: writes artifacts.kill_switch_event + direct-
        # resumes the parked live_wait → live_pause (no /wake, no new path). The
        # writer's own Fork-2 guard re-checks graph stage=="live".
        await kill_event_writer_fn(
            strategy_id,
            {"reason": RECONCILE_REASON, "action_taken": RECONCILE_ACTION, "metrics": metrics},
        )
        return "transitioned"
    except Exception as exc:  # noqa: BLE001 — one bad row must not abort the sweep
        logger.error(
            "reconcile: failed reconciling sid=%s stage=%s url=%s err=%s; continuing",
            strategy_id,
            stage,
            url,
            exc,
        )
        return "error"


async def reconcile_on_startup(
    *,
    kill_event_writer_fn: KillEventWriterFn,
    connect_fn: ConnectFn = _connect_app_db,
    client_factory: ClientFactory = _default_client_factory,
    record_kill_switch_event_fn: RecordKillFn = record_kill_switch_event,
) -> dict[str, Any]:
    """Scan ``strategy_registry`` and reconcile each container against reality.

    For every non-archived row carrying a ``freqtrade_api_url`` (the running
    paper/live containers), ping it: reachable → keep state; unreachable LIVE →
    write a ``kill_switch_events`` row (reason ``orchestrator_restart_no_freqtrade``)
    and drive the thread to ``live_pause`` via the 9f kill mechanism; unreachable
    non-live (paper) → log only.

    Wired into the FastAPI lifespan startup (after the saver/store + graph +
    kill writer are built, before serving). NEVER raises — a DB-list failure is
    swallowed and the lifespan keeps booting. Returns a summary dict
    (``checked`` / ``reachable`` / ``transitioned`` / ``skipped_non_live`` /
    ``errors``) for the startup log + tests.
    """
    try:
        rows = await _active_containers(connect_fn)
    except Exception as exc:  # noqa: BLE001 — a DB failure must not crash startup
        logger.warning("reconcile: could not list containers: %s; skipping", exc)
        return {
            "checked": 0,
            "reachable": [],
            "transitioned": [],
            "skipped_non_live": [],
            "errors": [],
        }

    reachable: list[str] = []
    transitioned: list[str] = []
    skipped_non_live: list[str] = []
    errors: list[str] = []

    # Sequential (not gathered): the sweep runs once at startup over ≤~20 rows,
    # and ordering keeps the summary deterministic for the log + tests. Each
    # _reconcile_one swallows its own failure, so the loop never aborts.
    for sid, stage, url in rows:
        bucket = await _reconcile_one(
            sid,
            stage,
            url,
            kill_event_writer_fn=kill_event_writer_fn,
            client_factory=client_factory,
            record_kill_switch_event_fn=record_kill_switch_event_fn,
        )
        {
            "reachable": reachable,
            "transitioned": transitioned,
            "skipped_non_live": skipped_non_live,
            "error": errors,
        }[bucket].append(sid)

    summary = {
        "checked": len(rows),
        "reachable": reachable,
        "transitioned": transitioned,
        "skipped_non_live": skipped_non_live,
        "errors": errors,
    }
    logger.info(
        "reconcile complete: checked=%d reachable=%d transitioned=%d "
        "skipped_non_live=%d errors=%d",
        summary["checked"],
        len(reachable),
        len(transitioned),
        len(skipped_non_live),
        len(errors),
    )
    return summary
