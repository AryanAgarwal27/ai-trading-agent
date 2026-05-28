"""APScheduler wiring — wake jobs, regime job, kill-switch poll (Stage 7f).

BRD references:
  - §13 Stage 7: "APScheduler with SQLAlchemyJobStore, wake job, regime job".
  - §5.5: paper thread woken every 6h → POST /threads/{tid}/wake.
  - §5.9: regime APScheduler job writes regime_log.
  - §11: kill-switch poll every 5 min (PLACEHOLDER here — the real
    /api/v1/profit polling + /stop lands in Stage 8 once live threads
    exist; before then there is nothing to poll).

Wake auth (SPEC §6, d6736ba locked decision): the per-thread wake job is
an httpx loopback POST to the orchestrator's own /threads/{tid}/wake
endpoint, carrying ``X-Operator-Token: $OPERATOR_TOKEN`` — the SAME auth
surface as /approve. The token is read from the environment at FIRE time,
never persisted into the jobstore (no secrets on disk in apscheduler_jobs).

Jobstore choice:
  - Production: ``SQLAlchemyJobStore`` against the app DB (DATABASE_URL),
    so a parked paper thread's wake survives an orchestrator restart
    (BRD §4). Per-thread wake jobs persist; the two recurring singletons
    (regime, kill-switch) are re-registered with ``replace_existing`` on
    every startup, which resets their next-run so they never misfire-spam.
  - Tests: set ``AIT_SCHEDULER_JOBSTORE=memory`` (conftest does this
    globally) so the test suite never reads/writes the production
    jobstore table and a leaked wake job can never fire httpx during a
    test run.

All job target functions are module-level (picklable) so the
SQLAlchemyJobStore can serialize the job reference by import path.
"""

from __future__ import annotations

import logging
import os

import httpx
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from orchestrator.subgraphs.paper import ScheduleWakeFn

logger = logging.getLogger(__name__)

# ─── Cadences (BRD §5.5, §5.9, §11) ────────────────────────────────────
WAKE_INTERVAL_HOURS = 6
REGIME_INTERVAL_HOURS = 1
KILL_SWITCH_INTERVAL_MINUTES = 5

# ─── Job-id conventions ─────────────────────────────────────────────────
WAKE_JOB_PREFIX = "wake:"
REGIME_JOB_ID = "regime_job"
KILL_SWITCH_JOB_ID = "kill_switch_poll"

# A missed wake (orchestrator down) within this window still fires on
# restart; older misses are skipped and caught by the next 6h cycle.
_WAKE_MISFIRE_GRACE_S = 3600


def _orchestrator_base_url() -> str:
    """Loopback base URL of the orchestrator's own FastAPI (BRD §15)."""
    host = os.environ.get("ORCHESTRATOR_HOST", "127.0.0.1")
    port = os.environ.get("ORCHESTRATOR_PORT", "8000")
    return f"http://{host}:{port}"


# ════════════════════════════════════════════════════════════════════════
# Job target functions (module-level → picklable for SQLAlchemyJobStore)
# ════════════════════════════════════════════════════════════════════════


async def _fire_wake(thread_id: str, base_url: str) -> None:
    """POST /threads/{thread_id}/wake on the orchestrator (loopback).

    Reads ``OPERATOR_TOKEN`` from the env at fire time (SPEC §6) — never
    persisted in the jobstore. Failures are logged, not raised: a wake
    that can't be delivered (orchestrator mid-restart, transient) is
    retried on the next 6h cycle; raising here would just spam the
    APScheduler error log without recovering anything.
    """
    token = os.environ.get("OPERATOR_TOKEN", "")
    headers = {"X-Operator-Token": token} if token else {}
    url = f"{base_url}/threads/{thread_id}/wake"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, headers=headers)
        if resp.status_code >= 400:
            logger.warning(
                "wake call non-2xx thread_id=%s status=%s body=%s",
                thread_id,
                resp.status_code,
                resp.text[:200],
            )
        else:
            logger.info("wake delivered thread_id=%s", thread_id)
    except httpx.HTTPError as exc:
        logger.warning("wake call transport error thread_id=%s exc=%s", thread_id, exc)


async def _fire_regime_job() -> None:
    """Classify the current BTC/USDT regime and insert a regime_log row.

    BRD §5.9 — the regime APScheduler job. Reads the most recent closes
    from the cached BTC/USDT 5m feather (SPEC §1 Q2 anchor pair),
    classifies via the vol+trend bucketing, and persists. Skips
    gracefully if the cache is absent (no market data downloaded yet) or
    too short to classify — a regime job that can't classify is a no-op,
    not an error.
    """
    import psycopg

    from orchestrator.tools.backtest_runner import SHARED_DATA_DIR
    from orchestrator.tools.regime import classify_regime, insert_regime_log

    feather_path = SHARED_DATA_DIR / "binance" / "BTC_USDT-5m.feather"
    if not feather_path.exists():
        logger.info("regime_job: no cached BTC/USDT feather at %s; skipping", feather_path)
        return

    import pyarrow.feather as feather

    table = feather.read_table(feather_path)  # type: ignore[no-untyped-call]
    closes = [float(c) for c in table.column("close").to_pylist()[-200:]]
    if len(closes) < 30:
        logger.info("regime_job: only %d closes (<30); skipping", len(closes))
        return

    regime, features = classify_regime(closes, timeframe_minutes=5)

    dsn = os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://", 1)
    conn = await psycopg.AsyncConnection.connect(dsn)
    try:
        await insert_regime_log(
            conn=conn, regime=regime, features=features, detector="vol_trend_v1"
        )
        await conn.commit()
    finally:
        await conn.close()
    logger.info("regime_job: inserted regime=%s", regime)


async def _fire_kill_switch_poll() -> None:
    """PLACEHOLDER kill-switch poll (BRD §11 — real impl in Stage 8).

    The real job will poll ``/api/v1/profit`` every 5 min for every LIVE
    thread and call ``/api/v1/stop`` directly on a drawdown / consecutive-
    loss breach, independent of the graph (BRD §11). No live threads
    exist before Stage 8, so this is a liveness no-op — logged at debug
    so the job's scheduling is observable without log spam.
    """
    logger.debug("kill_switch_poll placeholder tick (Stage 8 wires real polling)")


# ════════════════════════════════════════════════════════════════════════
# Scheduler construction + wiring
# ════════════════════════════════════════════════════════════════════════


def build_scheduler(*, jobstore_url: str | None = None) -> AsyncIOScheduler:
    """Build an ``AsyncIOScheduler`` with the appropriate jobstore.

    ``AIT_SCHEDULER_JOBSTORE=memory`` selects an in-memory store (tests).
    Otherwise a ``SQLAlchemyJobStore`` against ``jobstore_url`` (default
    ``DATABASE_URL``) so wakes survive restart (BRD §4).
    """
    mode = os.environ.get("AIT_SCHEDULER_JOBSTORE", "sqlalchemy")
    if mode == "memory":
        jobstores: dict[str, object] = {"default": MemoryJobStore()}
    else:
        url = jobstore_url or os.environ["DATABASE_URL"]
        jobstores = {"default": SQLAlchemyJobStore(url=url)}
    return AsyncIOScheduler(jobstores=jobstores, timezone="UTC")


def register_recurring_jobs(scheduler: AsyncIOScheduler) -> None:
    """Register the regime + kill-switch singletons (idempotent).

    ``replace_existing=True`` makes startup idempotent and resets each
    job's next-run to now+interval, so a persisted job from a prior run
    never misfire-fires on boot.
    """
    scheduler.add_job(
        _fire_regime_job,
        trigger="interval",
        hours=REGIME_INTERVAL_HOURS,
        id=REGIME_JOB_ID,
        replace_existing=True,
        coalesce=True,
        misfire_grace_time=600,
    )
    scheduler.add_job(
        _fire_kill_switch_poll,
        trigger="interval",
        minutes=KILL_SWITCH_INTERVAL_MINUTES,
        id=KILL_SWITCH_JOB_ID,
        replace_existing=True,
        coalesce=True,
        misfire_grace_time=60,
    )


def make_schedule_wake_fn(
    scheduler: AsyncIOScheduler, *, base_url: str | None = None
) -> ScheduleWakeFn:
    """Return the ``schedule_wake_fn`` the paper subgraph calls (7e seam).

    The returned coroutine registers a per-thread interval wake job
    (every 6h, id ``wake:<thread_id>``) whose target is the module-level
    :func:`_fire_wake`. ``replace_existing=True`` so a re-spawn / replay
    re-arms rather than duplicating the job.
    """
    resolved_base = base_url or _orchestrator_base_url()

    async def schedule_wake_fn(thread_id: str, strategy_id: str) -> None:
        scheduler.add_job(
            _fire_wake,
            trigger="interval",
            hours=WAKE_INTERVAL_HOURS,
            args=[thread_id, resolved_base],
            id=f"{WAKE_JOB_PREFIX}{thread_id}",
            replace_existing=True,
            coalesce=True,
            misfire_grace_time=_WAKE_MISFIRE_GRACE_S,
        )
        logger.info(
            "scheduled wake job thread_id=%s strategy_id=%s every %dh",
            thread_id,
            strategy_id,
            WAKE_INTERVAL_HOURS,
        )

    return schedule_wake_fn


async def shutdown_scheduler(scheduler: AsyncIOScheduler) -> None:
    """Stop the scheduler without blocking on in-flight jobs.

    ``wait=False`` so a long wake httpx call in flight doesn't stall
    FastAPI shutdown; the job is idempotent and will re-fire next cycle.
    """
    if scheduler.running:
        scheduler.shutdown(wait=False)
