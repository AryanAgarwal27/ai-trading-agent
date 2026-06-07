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
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import psycopg
from apscheduler.jobstores.base import JobLookupError
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from orchestrator.gates.thresholds import (
    DAILY_LOSS_LIMIT_PCT,
    KILL_SWITCH_CONSECUTIVE_LOSSES,
    KILL_SWITCH_DRAWDOWN,
)
from orchestrator.observability.events import publish_kill, record_kill_switch_event
from orchestrator.subgraphs.paper import ScheduleWakeFn
from orchestrator.tools.freqtrade_api import FreqtradeAPI, FreqtradeAPIError, FreqtradeCredentials
from orchestrator.tools.freqtrade_metrics import trailing_losses

logger = logging.getLogger(__name__)

# ─── Cadences (BRD §5.5, §5.9, §11) ────────────────────────────────────
WAKE_INTERVAL_HOURS = 6
# Live periodic re-eval cadence (Stage 9f). Deliberately the SAME 6h as paper —
# NOT a tighter cadence — because acute live safety is already covered out-of-band
# by the 5-min kill-switch poll + 15-min daily-loss job (BRD §11); the live-wake's
# only job is the SOFT periodic LLM re-eval (performance drift vs paper, regime
# mismatch → coordinator → live_pause). BRD §12 budgets ~360 live cycles / 90 days
# = 4/day = 6h; a 30-min cadence would be ~12x that LLM spend for marginal
# soft-drift latency the kill switch doesn't need. Env-overridable for ops tuning.
LIVE_WAKE_INTERVAL_HOURS = 6
REGIME_INTERVAL_HOURS = 1
KILL_SWITCH_INTERVAL_MINUTES = 5
DAILY_LOSS_INTERVAL_MINUTES = 15

# ─── Job-id conventions ─────────────────────────────────────────────────
WAKE_JOB_PREFIX = "wake:"
# Live-wake jobs get a SEPARATE id namespace from paper's ``wake:<thread_id>`` so
# the two never collide on a thread that graduated paper→live (and so cleanup can
# target the live job by strategy_id). Keyed on strategy_id (the cleanup API,
# unschedule_live_wake, takes strategy_id).
LIVE_WAKE_JOB_PREFIX = "live_wake:"
REGIME_JOB_ID = "regime_job"
KILL_SWITCH_JOB_ID = "kill_switch_poll"
DAILY_LOSS_JOB_ID = "daily_loss_poll"
SUPERVISOR_CRON_JOB_ID = "supervisor_cron"

# ─── Supervisor nightly cron (Stage 9d, BRD §5.1, §13 row 9) ────────────
# The supervisor cron job CLOSES OVER the graph + store (via the
# lifespan-built ``run_supervisor_fn`` partial), which are NOT picklable, so it
# cannot live in the SQLAlchemyJobStore. It gets a dedicated in-memory jobstore,
# re-registered on every startup (replace_existing). Default time 03:15 UTC —
# a low-activity global window; overridable via SUPERVISOR_CRON_HOUR /
# SUPERVISOR_CRON_MINUTE (the scheduler runs in UTC, so set your local-equivalent
# UTC time). Grace matches the low-frequency wake-job grace (3600), not the
# high-frequency poll graces (60–600): a nightly job tolerates a wider
# on-time-but-delayed window than a 5-minute poll.
SUPERVISOR_JOBSTORE = "supervisor_memory"
SUPERVISOR_CRON_HOUR = 3
SUPERVISOR_CRON_MINUTE = 15
SUPERVISOR_CRON_GRACE_S = 3600

# A missed wake (orchestrator down) within this window still fires on
# restart; older misses are skipped and caught by the next 6h cycle.
_WAKE_MISFIRE_GRACE_S = 3600

# Kill-switch REST timeout — tight (BRD §11): if Freqtrade is unresponsive,
# log loud and proceed; the kill switch must not block the job loop.
_KILL_REST_TIMEOUT_S = 5.0

# action_taken values written to kill_switch_events (BRD §5.8).
_ACTION_STOP = "POST /api/v1/stop"
_ACTION_STOP_TIMEOUT = "stop_call_timeout"
_ACTION_STOPBUY = "POST /api/v1/stopbuy"
_ACTION_STOPBUY_TIMEOUT = "stopbuy_call_timeout"

# Stop-action set the kill-switch idempotency guard checks against.
_STOP_ACTIONS = (_ACTION_STOP, _ACTION_STOP_TIMEOUT)


def _orchestrator_base_url() -> str:
    """Loopback base URL of the orchestrator's own FastAPI (BRD §15)."""
    host = os.environ.get("ORCHESTRATOR_HOST", "127.0.0.1")
    port = os.environ.get("ORCHESTRATOR_PORT", "8000")
    return f"http://{host}:{port}"


# ════════════════════════════════════════════════════════════════════════
# Job target functions (module-level → picklable for SQLAlchemyJobStore)
# ════════════════════════════════════════════════════════════════════════


async def _fire_wake(thread_id: str, base_url: str, kind: str = "paper_wait") -> None:
    """POST /threads/{thread_id}/wake?kind=<kind> on the orchestrator (loopback).

    Reads ``OPERATOR_TOKEN`` from the env at fire time (SPEC §6) — never
    persisted in the jobstore. Failures are logged, not raised: a wake
    that can't be delivered (orchestrator mid-restart, transient) is
    retried on the next cycle; raising here would just spam the
    APScheduler error log without recovering anything.

    ``kind`` (Stage 9f) declares which parked-interrupt kind this wake is for —
    ``"paper_wait"`` (default, so the Stage-7f paper wake jobs persisted with
    only ``args=[thread_id, base_url]`` keep working unchanged) or ``"live_wait"``
    (the live-wake jobs registered by :func:`make_schedule_live_wake_fn`). The
    /wake endpoint validates the parked interrupt matches ``kind`` and 409s
    otherwise, so a live-wake can never resume a paper park or vice versa.
    """
    token = os.environ.get("OPERATOR_TOKEN", "")
    headers = {"X-Operator-Token": token} if token else {}
    url = f"{base_url}/threads/{thread_id}/wake"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, headers=headers, params={"kind": kind})
        if resp.status_code >= 400:
            logger.warning(
                "wake call non-2xx thread_id=%s kind=%s status=%s body=%s",
                thread_id,
                kind,
                resp.status_code,
                resp.text[:200],
            )
        else:
            logger.info("wake delivered thread_id=%s kind=%s", thread_id, kind)
    except httpx.HTTPError as exc:
        logger.warning(
            "wake call transport error thread_id=%s kind=%s exc=%s", thread_id, kind, exc
        )


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


# ════════════════════════════════════════════════════════════════════════
# Kill switch + daily-loss (Stage 8f) — out-of-band, graph-independent
# ════════════════════════════════════════════════════════════════════════
#
# BRD §1.1 rule 7 + §11: these jobs poll Freqtrade REST directly and call
# /stop or /stopbuy on threshold breach. They do NOT wait for the graph to
# wake, route through the coordinator, or respect any LLM verdict. They are
# the HARD floor under the coordinator's SOFT safety layer (8d). The graph
# learns of a kill via artifacts.kill_switch_event, which 8g sets from the
# Redis subscription this job publishes to.

# Injection seams — defaults are the real DB/REST/Redis impls; tests pass stubs.
LiveStrategiesFn = Callable[[], Awaitable[list[tuple[str, str]]]]
RestClientFactory = Callable[[str], Any]
AlreadyFiredFn = Callable[[str], Awaitable[bool]]
RecordEventFn = Callable[..., Awaitable[int]]
PublishKillFn = Callable[[str, dict[str, Any]], Awaitable[None]]


def _libpq_dsn(url: str) -> str:
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


async def _connect_app_db() -> psycopg.AsyncConnection:
    return await psycopg.AsyncConnection.connect(_libpq_dsn(os.environ["DATABASE_URL"]))


async def _list_live_strategies() -> list[tuple[str, str]]:
    """Default ``list_live_fn`` — (strategy_id, api_url) for every live thread."""
    conn = await _connect_app_db()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT strategy_id, freqtrade_api_url FROM strategy_registry "
                "WHERE stage = 'live' AND freqtrade_api_url IS NOT NULL"
            )
            rows = await cur.fetchall()
    finally:
        await conn.close()
    return [(str(r[0]), str(r[1])) for r in rows]


async def _kill_already_fired(strategy_id: str) -> bool:
    """Fork 5d guard: has a /stop already fired THIS live run?

    True iff a stop-action ``kill_switch_events`` row exists with
    ``fired_at > strategy_registry.live_started_at`` — i.e. fired after the
    current live run began. A fresh live_spawn bumps ``live_started_at`` and
    un-suppresses. NULL ``live_started_at`` → comparison excludes the row →
    returns False → kill fires (safe direction). See DEFERRED.md D-5.
    """
    conn = await _connect_app_db()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT 1 FROM kill_switch_events kse "
                "JOIN strategy_registry sr ON kse.strategy_id = sr.strategy_id "
                "WHERE kse.strategy_id = %s "
                "AND kse.action_taken = ANY(%s) "
                "AND kse.fired_at > sr.live_started_at "
                "ORDER BY kse.fired_at DESC LIMIT 1",
                (strategy_id, list(_STOP_ACTIONS)),
            )
            row = await cur.fetchone()
    finally:
        await conn.close()
    return row is not None


async def _stopbuy_already_fired_24h(strategy_id: str) -> bool:
    """daily_loss guard: a stopbuy row in the last 24h (sliding window).

    stopbuy is a soft halt; a fresh calendar day legitimately re-evaluates, so
    the guard is a 24h sliding window rather than the live-run anchor.
    """
    conn = await _connect_app_db()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT 1 FROM kill_switch_events WHERE strategy_id = %s "
                "AND action_taken = %s AND fired_at > now() - interval '24 hours' "
                "ORDER BY fired_at DESC LIMIT 1",
                (strategy_id, _ACTION_STOPBUY),
            )
            row = await cur.fetchone()
    finally:
        await conn.close()
    return row is not None


def _default_rest_client(api_url: str) -> FreqtradeAPI:
    """Default ``rest_client_factory`` — a live-keyed REST client, tight timeout."""
    creds = FreqtradeCredentials(
        username="freqtrader",
        password=os.environ.get("BINANCE_LIVE_API_PASSWORD", ""),
    )
    return FreqtradeAPI(base_url=api_url, credentials=creds, timeout_s=_KILL_REST_TIMEOUT_S)


def _kill_reason(max_drawdown: float, consecutive_losses: int) -> str | None:
    """Return the breach-reason string, or None if no threshold is breached."""
    if max_drawdown >= KILL_SWITCH_DRAWDOWN:
        return f"drawdown_{int(KILL_SWITCH_DRAWDOWN * 100)}pct_exceeded"
    if consecutive_losses >= KILL_SWITCH_CONSECUTIVE_LOSSES:
        return f"consecutive_losses_{KILL_SWITCH_CONSECUTIVE_LOSSES}_exceeded"
    return None


def _today_rel_profit(daily_resp: dict[str, Any]) -> float:
    """Today's relative profit from a ``/api/v1/daily`` response.

    Freqtrade 2026.4 /api/v1/daily returns ``data: [{date, abs_profit,
    rel_profit, trade_count, ...}]`` (most recent first); field-name
    verification is deferred to the AIT_RUN_REAL_LIVE_SPAWN_TESTS smoke per the
    8a stub-only-CI stance. Defensive: missing/empty data → 0.0 (no breach).
    """
    data = daily_resp.get("data", []) if isinstance(daily_resp, dict) else []
    if not data or not isinstance(data[0], dict):
        return 0.0
    return float(data[0].get("rel_profit", 0.0) or 0.0)


async def kill_switch_poll_job(
    *,
    list_live_fn: LiveStrategiesFn | None = None,
    rest_client_factory: RestClientFactory | None = None,
    already_fired_fn: AlreadyFiredFn | None = None,
    record_event_fn: RecordEventFn | None = None,
    publish_fn: PublishKillFn | None = None,
) -> None:
    """Out-of-band kill switch (BRD §1.1 rule 7, §11). Every 5 min.

    For each stage='live' strategy: poll /profit + /trades; on drawdown ≥
    KILL_SWITCH_DRAWDOWN or consecutive losses ≥ KILL_SWITCH_CONSECUTIVE_LOSSES,
    POST /stop directly (tight timeout), write a kill_switch_events row, and
    publish to Redis. A strategy already killed this live run (Fork 5d guard) is
    skipped. One strategy's failure never aborts the loop.
    """
    list_live = list_live_fn or _list_live_strategies
    factory = rest_client_factory or _default_rest_client
    already_fired = already_fired_fn or _kill_already_fired
    record_event = record_event_fn or record_kill_switch_event
    publish = publish_fn or publish_kill

    for strategy_id, api_url in await list_live():
        try:
            if await already_fired(strategy_id):
                logger.info("kill_switch already fired for %s, awaiting graph routing", strategy_id)
                continue

            action_taken = _ACTION_STOP
            reason: str | None = None
            max_dd = 0.0
            consecutive = 0
            profit: dict[str, Any] = {}

            client = factory(api_url)
            async with client:
                profit_resp = await client.profit()
                trades_resp = await client.trades(limit=500)
                profit = profit_resp if isinstance(profit_resp, dict) else {}
                trades = trades_resp.get("trades", []) if isinstance(trades_resp, dict) else []
                max_dd = float(profit.get("max_drawdown", 0.0) or 0.0)
                consecutive = trailing_losses(trades)
                reason = _kill_reason(max_dd, consecutive)
                if reason is None:
                    continue
                # BREACH — POST /stop directly. On a hang, log loud and STILL
                # record + publish (the graph must learn of the breach even if
                # /stop didn't confirm — the operator reviews at live_pause).
                try:
                    await client.stop()
                except (FreqtradeAPIError, httpx.HTTPError) as exc:
                    logger.error(
                        "KILL SWITCH /stop unresponsive sid=%s reason=%s exc=%s",
                        strategy_id,
                        reason,
                        exc,
                    )
                    action_taken = _ACTION_STOP_TIMEOUT

            fired_at = datetime.now(UTC).isoformat()
            await record_event(
                strategy_id=strategy_id,
                reason=reason,
                metrics={
                    "max_drawdown": max_dd,
                    "consecutive_losses": consecutive,
                    "profit": profit,
                },
                action_taken=action_taken,
            )
            await publish(
                strategy_id,
                {
                    "reason": reason,
                    "fired_at": fired_at,
                    "action_taken": action_taken,
                    "metrics_summary": {
                        "max_drawdown": max_dd,
                        "consecutive_losses": consecutive,
                    },
                },
            )
            logger.warning(
                "KILL SWITCH FIRED sid=%s reason=%s action=%s",
                strategy_id,
                reason,
                action_taken,
            )
        except Exception as exc:  # noqa: BLE001 — one strategy must not abort the loop
            logger.error("kill_switch_poll error sid=%s exc=%s; continuing", strategy_id, exc)


async def daily_loss_job(
    *,
    list_live_fn: LiveStrategiesFn | None = None,
    rest_client_factory: RestClientFactory | None = None,
    already_fired_fn: AlreadyFiredFn | None = None,
    record_event_fn: RecordEventFn | None = None,
) -> None:
    """Daily-loss soft halt (BRD §11). Every 15 min.

    For each stage='live' strategy: read the calendar-day /api/v1/daily bucket
    (D-3 approximation of rolling-24h); if today's relative profit ≤
    -DAILY_LOSS_LIMIT_PCT, POST /stopbuy (block new entries, let opens run) and
    write a kill_switch_events row. NO Redis publish — stopbuy is a soft halt,
    not a kill. Skip a strategy with a stopbuy in the last 24h.
    """
    list_live = list_live_fn or _list_live_strategies
    factory = rest_client_factory or _default_rest_client
    already_fired = already_fired_fn or _stopbuy_already_fired_24h
    record_event = record_event_fn or record_kill_switch_event

    for strategy_id, api_url in await list_live():
        try:
            if await already_fired(strategy_id):
                logger.info(
                    "daily_loss stopbuy already fired for %s within 24h, skipping", strategy_id
                )
                continue

            today_rel = 0.0
            daily: dict[str, Any] = {}
            action_taken = _ACTION_STOPBUY

            client = factory(api_url)
            async with client:
                # D-3: calendar-day bucket as a deliberate rolling-24h approximation.
                daily = await client.daily(timescale=1)
                today_rel = _today_rel_profit(daily)
                if today_rel > -DAILY_LOSS_LIMIT_PCT:
                    continue
                try:
                    await client.stopbuy()
                except (FreqtradeAPIError, httpx.HTTPError) as exc:
                    logger.error("daily_loss /stopbuy unresponsive sid=%s exc=%s", strategy_id, exc)
                    action_taken = _ACTION_STOPBUY_TIMEOUT

            await record_event(
                strategy_id=strategy_id,
                reason=f"daily_loss_{int(DAILY_LOSS_LIMIT_PCT * 100)}pct_exceeded",
                metrics={"today_rel_profit": today_rel, "daily": daily},
                action_taken=action_taken,
            )
            logger.warning("DAILY LOSS stopbuy sid=%s today_rel=%.4f", strategy_id, today_rel)
        except Exception as exc:  # noqa: BLE001 — one strategy must not abort the loop
            logger.error("daily_loss_job error sid=%s exc=%s; continuing", strategy_id, exc)


# ════════════════════════════════════════════════════════════════════════
# Supervisor nightly cron (Stage 9d)
# ════════════════════════════════════════════════════════════════════════
#
# BRD §5.1 + §13 row 9: the supervisor runs nightly. This job is the cron
# trigger; the event-driven trigger (on thread completion) lands in Stage 9e.
# The graph + store the runner needs are bound at lifespan time into
# ``run_supervisor_fn`` (a partial), so this job only needs to open a fresh
# app-DB connection per run — the SAME per-call connection pattern as
# kill_switch_poll_job (DB connections are not held across the idle day).

# The lifespan-bound runner: ``partial(run_supervisor, graph, store)``, called
# as ``run_supervisor_fn(conn, trigger="cron")``. Duck-typed (Awaitable[Any])
# to keep scheduler.py decoupled from orchestrator.supervisor.
SupervisorRunFn = Callable[..., Awaitable[Any]]


async def supervisor_cron_job(*, run_supervisor_fn: SupervisorRunFn) -> None:
    """Nightly supervisor invocation (BRD §5.1, §13 row 9).

    Opens a fresh app-DB connection (per-call, like kill_switch_poll_job), runs
    the supervisor at ``trigger="cron"`` via the lifespan-bound runner (graph +
    store already bound into ``run_supervisor_fn``), logs the decision summary,
    and closes the connection. Any exception is logged and SWALLOWED — one bad
    nightly run must not crash the scheduler or skip subsequent nights. The
    runner itself owns its transaction (commit/rollback); this job only manages
    the connection lifecycle.
    """
    conn = await _connect_app_db()
    try:
        decision = await run_supervisor_fn(conn, trigger="cron")
        actions = getattr(decision, "actions", []) or []
        rationale = str(getattr(decision, "overall_rationale", ""))
        logger.info(
            "supervisor cron: %d action(s); rationale=%s",
            len(actions),
            rationale[:200],
        )
    except Exception as exc:  # noqa: BLE001 — one bad run must not kill the scheduler
        logger.error(
            "supervisor cron run failed: %s; swallowing (next nightly fire unaffected)", exc
        )
    finally:
        await conn.close()


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
        kill_switch_poll_job,
        trigger="interval",
        minutes=KILL_SWITCH_INTERVAL_MINUTES,
        id=KILL_SWITCH_JOB_ID,
        replace_existing=True,
        coalesce=True,
        misfire_grace_time=60,
    )
    scheduler.add_job(
        daily_loss_job,
        trigger="interval",
        minutes=DAILY_LOSS_INTERVAL_MINUTES,
        id=DAILY_LOSS_JOB_ID,
        replace_existing=True,
        coalesce=True,
        misfire_grace_time=120,
    )


def register_supervisor_cron(
    scheduler: AsyncIOScheduler,
    *,
    run_supervisor_fn: SupervisorRunFn,
    hour: int | None = None,
    minute: int | None = None,
) -> None:
    """Register the nightly supervisor cron job (Stage 9d; idempotent).

    Separate from :func:`register_recurring_jobs` because this job CLOSES OVER
    the graph + store (via ``run_supervisor_fn``, the lifespan partial), which
    are not picklable — so it lives in a dedicated **in-memory** jobstore rather
    than the SQLAlchemyJobStore the persisted wake/interval jobs use. Called
    from the FastAPI lifespan after the graph is built. ``replace_existing=True``
    + the memory store make re-registration on every startup idempotent and
    recompute the next cron fire, so a restart never replays a missed run.

    Time defaults to ``SUPERVISOR_CRON_HOUR``:``SUPERVISOR_CRON_MINUTE`` (03:15
    UTC), overridable via the matching env vars (the scheduler is UTC — set
    your local-equivalent UTC time) or the ``hour`` / ``minute`` params (tests).
    ``max_instances=1`` so a slow run (LLM calls of 15–25s, occasionally more)
    never overlaps the next fire; ``coalesce=True`` collapses a backlog to one.
    """
    resolved_hour = (
        hour
        if hour is not None
        else int(os.environ.get("SUPERVISOR_CRON_HOUR", SUPERVISOR_CRON_HOUR))
    )
    resolved_minute = (
        minute
        if minute is not None
        else int(os.environ.get("SUPERVISOR_CRON_MINUTE", SUPERVISOR_CRON_MINUTE))
    )

    # Dedicated in-memory jobstore for the non-picklable closure job.
    # ``add_jobstore`` raises ValueError if the alias already exists (a
    # re-registration against the SAME scheduler instance) — guard for
    # idempotency. The trigger inherits the scheduler's UTC timezone.
    try:
        scheduler.add_jobstore(MemoryJobStore(), SUPERVISOR_JOBSTORE)
    except ValueError:
        pass

    scheduler.add_job(
        supervisor_cron_job,
        trigger=CronTrigger(hour=resolved_hour, minute=resolved_minute),
        kwargs={"run_supervisor_fn": run_supervisor_fn},
        id=SUPERVISOR_CRON_JOB_ID,
        jobstore=SUPERVISOR_JOBSTORE,
        replace_existing=True,
        coalesce=True,
        max_instances=1,
        misfire_grace_time=SUPERVISOR_CRON_GRACE_S,
    )
    logger.info(
        "registered supervisor cron job id=%s at %02d:%02d UTC (jobstore=%s)",
        SUPERVISOR_CRON_JOB_ID,
        resolved_hour,
        resolved_minute,
        SUPERVISOR_JOBSTORE,
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


# ─── Live-wake (Stage 9f, D-6 periodic re-eval path) ────────────────────
# (strategy_id) -> remove that strategy's live-wake job. Injected into
# live_archive + aretire_strategy so an archived live thread's recurring wake
# job is cancelled rather than left to fire forever and harmlessly 409.
UnscheduleWakeFn = Callable[[str], Awaitable[None]]


def make_schedule_live_wake_fn(
    scheduler: AsyncIOScheduler, *, base_url: str | None = None
) -> ScheduleWakeFn:
    """Return the live ``schedule_wake_fn`` (Stage 9f) — symmetric to paper's.

    Same ``ScheduleWakeFn`` shape ``(thread_id, strategy_id)`` and same
    module-level :func:`_fire_wake` target as :func:`make_schedule_wake_fn`, but
    registers a SEPARATE interval job ``live_wake:<strategy_id>`` firing every
    ``LIVE_WAKE_INTERVAL_HOURS`` (6h — see the constant's note) and passing
    ``kind="live_wait"`` so the /wake endpoint routes the resume into the live
    subgraph's ``live_wait`` park (not paper's). Persisted in the SQLAlchemy
    jobstore (the default) so a parked live thread keeps being woken across an
    orchestrator restart (BRD §4). ``replace_existing=True`` → re-spawn / replay
    re-arms rather than duplicating. Job-id is keyed on ``strategy_id`` so
    :func:`make_unschedule_live_wake_fn` can cancel it at archive.
    """
    resolved_base = base_url or _orchestrator_base_url()

    async def schedule_live_wake_fn(thread_id: str, strategy_id: str) -> None:
        scheduler.add_job(
            _fire_wake,
            trigger="interval",
            hours=LIVE_WAKE_INTERVAL_HOURS,
            args=[thread_id, resolved_base],
            kwargs={"kind": "live_wait"},
            id=f"{LIVE_WAKE_JOB_PREFIX}{strategy_id}",
            replace_existing=True,
            coalesce=True,
            misfire_grace_time=_WAKE_MISFIRE_GRACE_S,
        )
        logger.info(
            "scheduled LIVE wake job strategy_id=%s thread_id=%s every %dh",
            strategy_id,
            thread_id,
            LIVE_WAKE_INTERVAL_HOURS,
        )

    return schedule_live_wake_fn


def make_unschedule_live_wake_fn(scheduler: AsyncIOScheduler) -> UnscheduleWakeFn:
    """Return the live-wake cleanup fn (Stage 9f) — cancels ``live_wake:<sid>``.

    Called at every live-archive path (``live_archive``, supervisor
    ``aretire_strategy`` when the thread was live) so an archived live thread's
    recurring wake job is removed. Idempotent: removing a non-existent job (never
    scheduled, already removed, or a startup-replayed orphan already gone) is a
    swallowed no-op — a missed cleanup only costs harmless 409s on the next fire.
    """

    async def unschedule_live_wake(strategy_id: str) -> None:
        job_id = f"{LIVE_WAKE_JOB_PREFIX}{strategy_id}"
        try:
            scheduler.remove_job(job_id)
            logger.info("unscheduled LIVE wake job strategy_id=%s", strategy_id)
        except JobLookupError:
            logger.debug("unschedule_live_wake: no job %s (already gone)", job_id)

    return unschedule_live_wake


def make_unschedule_wake_fn(scheduler: AsyncIOScheduler) -> UnscheduleWakeFn:
    """Return the PAPER-wake cleanup fn (D-12) — cancels ``wake:<thread_id>``.

    Symmetric to :func:`make_unschedule_live_wake_fn`. Called at the paper
    ``archive`` terminal node (the sink for both the kill/reject teardown path and
    the spawn-failure path) so an archived paper thread's recurring wake job is
    removed rather than left to fire every 6h and harmlessly 409. The paper wake
    job (:func:`make_schedule_wake_fn`) is keyed on ``thread_id``, which is
    ``strategy_<strategy_id>`` by the spawn convention (``aspawn_strategy``), so
    the job id is reconstructed from the strategy_id. Idempotent: removing a
    non-existent job (the spawn-failure path never scheduled one, or it was
    already removed / replayed away) is a swallowed no-op.
    """

    async def unschedule_wake(strategy_id: str) -> None:
        job_id = f"{WAKE_JOB_PREFIX}strategy_{strategy_id}"
        try:
            scheduler.remove_job(job_id)
            logger.info("unscheduled paper wake job strategy_id=%s", strategy_id)
        except JobLookupError:
            logger.debug("unschedule_wake: no job %s (already gone)", job_id)

    return unschedule_wake


async def shutdown_scheduler(scheduler: AsyncIOScheduler) -> None:
    """Stop the scheduler without blocking on in-flight jobs.

    ``wait=False`` so a long wake httpx call in flight doesn't stall
    FastAPI shutdown; the job is idempotent and will re-fire next cycle.
    """
    if scheduler.running:
        scheduler.shutdown(wait=False)
