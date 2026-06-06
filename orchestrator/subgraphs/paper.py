"""Stage 7 paper subgraph (BRD §5.5) — paper_spawn node only.

This 7c commit ships ONLY ``paper_spawn``. The other nodes
(``schedule_wake``, ``paper_wait``, ``paper_monitor``,
``divergence_check``, ``live_gate``, ``paper_teardown``) land in 7e
once the spawn + monitor cycle has a real Freqtrade container to
talk to.

``paper_spawn`` is the first node in the paper subgraph. Pipeline:

1. Derive ``stake_amount`` from ``state.params`` (or fall back to
   ``LIVE_CAPITAL_CAP_USD / MAX_OPEN_TRADES`` per BRD §10).
2. Query ``strategy_registry`` for ports already assigned to active
   paper threads; call :func:`next_free_paper_port` for the next free.
3. **Upsert the registry row BEFORE calling spawn.** This is the
   orphan-container-prevention contract: if spawn succeeds but the
   registry write fails, the orchestrator loses track of a live
   container that's placing dry-run orders. Order matters.
4. Call ``spawn_container_fn`` (injected for tests; defaults to the
   real :func:`spawn_paper_container`). Catch ``PaperSpawnTimeout``
   and any other exception, return ``{stage: "archived", ...}`` —
   never raise. A spawn failure should archive the thread cleanly,
   not crash the graph thread.
5. On success: UPDATE the registry row with the resolved API URL.
6. Return state update including ``artifacts.paper_started_at`` —
   this is the 30-day clock SPEC §6 (d6736ba) locked. The
   ``divergence_check`` node (7e) enforces
   ``elapsed_days >= MIN_PAPER_DAYS`` against this timestamp before
   the monitor can vote "advance".
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from operator import add
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict
from urllib.parse import urlparse

import psycopg
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from orchestrator.agents.monitors import (
    _VOTE_VERDICT,
    PaperMonitorContext,
    run_paper_monitor,
)
from orchestrator.gates import thresholds
from orchestrator.gates.hitl import build_interrupt_payload
from orchestrator.gates.thresholds import LIVE_CAPITAL_CAP_USD, MAX_OPEN_TRADES
from orchestrator.observability.events import publish_gate_pending, publish_thread_completed
from orchestrator.observability.log import get_logger
from orchestrator.state import AgentVote, StrategyState
from orchestrator.tools.freqtrade_api import FreqtradeAPI, FreqtradeCredentials
from orchestrator.tools.freqtrade_lifecycle import (
    WORKERS_ROOT,
    PaperSpawnTimeout,
    next_free_paper_port,
    spawn_paper_container,
    stop_paper_container,
)

logger = logging.getLogger(__name__)

# BaseCheckpointSaver is generic on its serializer; accept any concrete
# saver (InMemorySaver for tests, AsyncPostgresSaver for production).
CheckpointSaver = BaseCheckpointSaver[Any]

# Injection seam — same shape as Stage 5's ``critic_fn`` and Stage 4's
# ``risk_analyst_fn``. Tests pass a stub; production uses the default.
SpawnContainerFn = Callable[..., Awaitable[str]]
# 7e injection seams. paper_monitor_fn defaults to the real Haiku agent
# (run_paper_monitor); build_context_fn defaults to fetching live metrics
# from the container; schedule_wake_fn defaults to a no-op until 7f wires
# APScheduler; stop_container_fn defaults to the 7b teardown helper.
PaperMonitorFn = Callable[[PaperMonitorContext], Awaitable[Any]]
# Accepts a Mapping so both PaperState and StrategyState (TypedDicts) pass
# — TypedDict is invariantly NOT a dict[str, Any] for mypy, but it IS a
# Mapping[str, Any].
BuildContextFn = Callable[[Mapping[str, Any]], Awaitable[PaperMonitorContext]]
ScheduleWakeFn = Callable[[str, str], Awaitable[None]]
StopContainerFn = Callable[[str], Awaitable[None]]
# D-9: () -> current live-strategy count (the capital axis, BRD §10). live_gate
# reads it BEFORE offering HITL approval; injected so tests drive the slot-full /
# slot-free paths hermetically. Default reads aget_portfolio_snapshot()["live"].
LiveCountFn = Callable[[], Awaitable[int]]


# ───────────────────────── db helpers ─────────────────────────


def _libpq_dsn(sqlalchemy_url: str) -> str:
    """Strip the ``postgresql+psycopg://`` SQLAlchemy prefix for libpq."""
    return sqlalchemy_url.replace("postgresql+psycopg://", "postgresql://", 1)


async def _connect_app_db() -> psycopg.AsyncConnection:
    """Open a fresh async psycopg connection to the app DB.

    Same shape as :func:`orchestrator.observability.events._connect_app_db`.
    Not reused from there to keep the paper subgraph's import surface
    independent of observability — the two modules ship and evolve at
    different cadences.
    """
    return await psycopg.AsyncConnection.connect(_libpq_dsn(os.environ["DATABASE_URL"]))


async def _default_live_count() -> int:
    """Current ``stage="live"`` count via ``aget_portfolio_snapshot()["live"]``.

    The D-9 live-cap read at ``live_gate`` (BRD §10, ``MAX_CONCURRENT_LIVE_STRATEGIES``).
    Opens a short-lived app-DB connection and reuses the supervisor's canonical
    snapshot so the count matches what the supervisor itself reasons on. Lazy
    import avoids a paper→supervisor module-load edge.
    """
    from orchestrator.supervisor import aget_portfolio_snapshot

    conn = await _connect_app_db()
    try:
        snapshot = await aget_portfolio_snapshot(conn)
        return int(snapshot["live"])
    finally:
        await conn.close()


def _used_ports_from_urls(urls: list[Any]) -> list[int]:
    """Extract port numbers from ``freqtrade_api_url`` row values.

    Tolerant of None and malformed URLs — a row without a parseable
    port simply doesn't contribute to the in-use set. The allocator
    is the authoritative source for collision-avoidance; this is a
    best-effort prune of the search space.
    """
    ports: list[int] = []
    for url in urls:
        if not isinstance(url, str):
            continue
        try:
            parsed = urlparse(url)
            if parsed.port is not None:
                ports.append(parsed.port)
        except ValueError:
            continue
    return ports


def _thread_id_for(config: RunnableConfig, strategy_id: str) -> str:
    """Return the thread_id from config, or the canonical fallback.

    LangGraph passes thread_id via ``config["configurable"]["thread_id"]``.
    Tests that invoke the node directly may omit this — falling back to
    ``strategy_<id>`` matches the convention used elsewhere in the
    codebase (BRD §5.1).
    """
    configurable = config.get("configurable") or {}
    thread_id = configurable.get("thread_id")
    if isinstance(thread_id, str) and thread_id:
        return thread_id
    return f"strategy_{strategy_id}"


async def _upsert_registry_row(
    conn: psycopg.AsyncConnection,
    state: StrategyState,
    thread_id: str,
) -> None:
    """Upsert the strategy_registry row for ``state`` with ``stage='paper'``.

    On conflict, only ``stage`` and ``last_updated`` are touched — the
    immutable identity fields (thread_id, name, template, pairs,
    timeframe) are not overwritten. ``freqtrade_api_url`` is left NULL
    here and filled by a follow-up UPDATE after spawn returns.

    Idempotent: re-running the node (replay after interrupt, restart
    after crash, smoke test re-run) leaves the registry in the same
    state regardless of prior runs.
    """
    import json as _json  # local import keeps top-level surface tidy

    async with conn.cursor() as cur:
        await cur.execute(
            """
            INSERT INTO strategy_registry
              (strategy_id, thread_id, name, template, stage, pairs,
               timeframe, started_at, last_updated)
            VALUES (%s, %s, %s, %s, 'paper', %s, %s, now(), now())
            ON CONFLICT (strategy_id) DO UPDATE SET
              stage = 'paper',
              last_updated = now()
            """,
            (
                state["strategy_id"],
                thread_id,
                state["name"],
                state["template"],
                _json.dumps(state["pairs"]),
                state["timeframe"],
            ),
        )


# ───────────────────────── node ─────────────────────────


async def paper_spawn(
    state: StrategyState,
    config: RunnableConfig,
    *,
    spawn_container_fn: SpawnContainerFn | None = None,
) -> dict[str, Any]:
    """First node in the paper subgraph — see module docstring for contract.

    Returns a state-update dict (LangGraph node convention). Never
    raises on spawn failure; failures route to ``stage="archived"``
    with a descriptive ``failure_reason``.
    """
    spawn_fn: SpawnContainerFn = spawn_container_fn or spawn_paper_container

    strategy_id = state["strategy_id"]
    get_logger("paper_spawn").info("enter", payload={"strategy_id": strategy_id})
    pairs = state["pairs"]
    params = state.get("params") or {}
    artifacts = state.get("artifacts") or {}

    # Stake amount: explicit param wins; otherwise BRD §10 default.
    default_stake = LIVE_CAPITAL_CAP_USD / MAX_OPEN_TRADES
    stake_amount = float(params.get("stake_amount", default_stake))

    # Generated strategy path is written by Stage 5's generator node
    # into state.artifacts. Missing it is a research-subgraph contract
    # violation — archive cleanly with a diagnostic.
    strategy_path_str = artifacts.get("generated_strategy_path")
    if not strategy_path_str:
        logger.error(
            "paper_spawn missing generated_strategy_path strategy_id=%s",
            strategy_id,
        )
        return {
            "stage": "archived",
            "failure_reason": (
                "paper_spawn_missing_strategy_path: state.artifacts has no "
                "'generated_strategy_path' — research subgraph must run first"
            ),
        }
    strategy_path = Path(strategy_path_str)

    thread_id = _thread_id_for(config, strategy_id)

    # 2 + 3: query used ports, allocate, upsert row BEFORE spawn.
    conn = await _connect_app_db()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT freqtrade_api_url FROM strategy_registry "
                "WHERE freqtrade_api_url IS NOT NULL"
            )
            rows = await cur.fetchall()
        used = _used_ports_from_urls([r[0] for r in rows])
        port = next_free_paper_port(used)

        await _upsert_registry_row(conn, state, thread_id)
        await conn.commit()
    finally:
        await conn.close()

    logger.info(
        "paper_spawn allocated port=%d strategy_id=%s thread_id=%s",
        port,
        strategy_id,
        thread_id,
    )

    # 4. Call spawn — both branches archive cleanly, neither raises.
    try:
        api_url = await spawn_fn(
            strategy_id=strategy_id,
            pair_whitelist=pairs,
            stake_amount=stake_amount,
            strategy_module_path=strategy_path,
            port=port,
        )
    except PaperSpawnTimeout as exc:
        logger.error(
            "paper_spawn timeout strategy_id=%s port=%d exc=%s",
            strategy_id,
            port,
            exc,
        )
        return {
            "stage": "archived",
            "failure_reason": f"paper_spawn_timeout: {strategy_id}",
        }
    except Exception as exc:  # noqa: BLE001 — any spawn failure archives cleanly
        logger.error(
            "paper_spawn failed strategy_id=%s port=%d exc_type=%s exc=%s",
            strategy_id,
            port,
            type(exc).__name__,
            exc,
        )
        type_name = type(exc).__name__
        return {
            "stage": "archived",
            "failure_reason": f"paper_spawn_failed: {type_name}: {str(exc)[:200]}",
        }

    # 5. UPDATE registry with the URL.
    conn = await _connect_app_db()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE strategy_registry SET freqtrade_api_url = %s, "
                "last_updated = now() WHERE strategy_id = %s",
                (api_url, strategy_id),
            )
        await conn.commit()
    finally:
        await conn.close()

    # 6. State update — paper_started_at is the 30-day clock anchor
    # SPEC §6 (d6736ba) locked. divergence_check enforces
    # elapsed_days >= MIN_PAPER_DAYS against this timestamp.
    #
    # PRESERVE an existing paper_started_at: the 30-day clock starts at
    # the FIRST spawn and must not reset if paper_spawn re-runs (crash
    # restart / replay / re-spawn after a container died). Resetting the
    # clock would silently extend the paper period past 30 days and
    # defeat the gate. Only stamp now() when the clock isn't already set.
    started_at_iso = artifacts.get("paper_started_at") or datetime.now(UTC).isoformat()
    worker_dir = WORKERS_ROOT / strategy_id
    return {
        "stage": "paper",
        "freqtrade_api_url": api_url,
        "freqtrade_userdir": str(worker_dir),
        "freqtrade_process_id": f"paper-{strategy_id}",
        "artifacts": {
            **artifacts,
            "paper_started_at": started_at_iso,
        },
    }


# ════════════════════════════════════════════════════════════════════════
# Paper subgraph state (Stage 7e)
# ════════════════════════════════════════════════════════════════════════


class PaperState(TypedDict, total=False):
    """Workspace state for the paper subgraph.

    ``total=False`` so the parent graph can hand a partial dict; nodes
    populate fields as they run. ``agent_votes`` is the BRD §6.3 reducer
    field — paper_monitor appends one vote per wake, so it MUST use
    ``Annotated[..., add]`` or successive wakes would overwrite.
    """

    # Identity / inputs
    strategy_id: str
    name: str
    template: str
    pairs: list[str]
    timeframe: str
    params: dict[str, Any]

    # Lifecycle
    stage: str

    # Agent vote trail (reducer — paper_monitor appends per wake)
    agent_votes: Annotated[list[AgentVote], add]

    # Gate verdicts + execution handles
    gate_decisions: dict[str, Any]
    freqtrade_api_url: str | None
    freqtrade_userdir: str | None
    freqtrade_process_id: str | None
    artifacts: dict[str, Any]

    # Terminal
    failure_reason: str


# ════════════════════════════════════════════════════════════════════════
# Helpers shared by the paper-subgraph nodes
# ════════════════════════════════════════════════════════════════════════


def _elapsed_paper_days(state: Mapping[str, Any]) -> float:
    """Days since ``artifacts.paper_started_at`` (the SPEC-locked clock).

    Returns 0.0 if the timestamp is missing or unparseable — divergence_check
    treats 0.0 as "below MIN_PAPER_DAYS", which is the safe default (it will
    not let a strategy advance to live without a real elapsed-time anchor).
    """
    artifacts = state.get("artifacts") or {}
    started = artifacts.get("paper_started_at")
    if not started:
        return 0.0
    try:
        started_dt = datetime.fromisoformat(started)
    except (ValueError, TypeError):
        return 0.0
    return (datetime.now(UTC) - started_dt).total_seconds() / 86400.0


def _trailing_losses(trades: list[dict[str, Any]]) -> int:
    """Count the trailing run of losing closed trades (most-recent-first).

    Freqtrade ``/trades`` returns trades oldest-first, so we walk in
    reverse. A non-dict or missing ``profit_ratio`` ends the run.
    """
    count = 0
    for t in reversed(trades):
        if not isinstance(t, dict):
            break
        pr = t.get("profit_ratio")
        if pr is None:
            break
        if float(pr) < 0:
            count += 1
        else:
            break
    return count


# ════════════════════════════════════════════════════════════════════════
# schedule_wake (Stage 7e — APScheduler wiring lands in 7f)
# ════════════════════════════════════════════════════════════════════════


async def _noop_schedule_wake(thread_id: str, strategy_id: str) -> None:
    """Default schedule_wake: do nothing.

    7f replaces this with the real APScheduler interval-job registration
    (every 6h, POST /threads/{tid}/wake). Until then the no-op lets the
    subgraph topology and tests run without a scheduler — a paper thread
    parks at paper_wait and is driven by an explicit Command(resume=...)
    (the test wake, or 7f's scheduler call in production).
    """
    return None


async def schedule_wake_node(
    state: PaperState,
    config: RunnableConfig,
    *,
    schedule_wake_fn: ScheduleWakeFn | None = None,
) -> dict[str, Any]:
    """Register the 6-hour wake job for this thread (BRD §5.5).

    Side-effect-only node — registers the APScheduler job and returns no
    state change. Safe to re-run (the real 7f registration is idempotent
    on job id = thread_id).
    """
    get_logger("schedule_wake").info("enter", payload={"strategy_id": state.get("strategy_id")})
    fn = schedule_wake_fn or _noop_schedule_wake
    thread_id = _thread_id_for(config, state["strategy_id"])
    await fn(thread_id, state["strategy_id"])
    return {}


# ════════════════════════════════════════════════════════════════════════
# paper_wait (dynamic interrupt — parks until a wake)
# ════════════════════════════════════════════════════════════════════════


def paper_wait(state: PaperState) -> dict[str, Any]:
    """Park the thread until a wake fires (BRD §5.5).

    A bare ``interrupt()`` with a wake payload. Unlike the HITL gates
    (paper_gate / live_gate) there is NO dashboard card for a wake-park,
    so there is no publish here — the only contract that applies is
    "no side effects on replay", which a bare interrupt satisfies.

    The resume value (whatever the wake caller sends via
    ``Command(resume=...)``) is intentionally ignored; the wake's only
    job is to un-park the thread so it proceeds to paper_monitor. In
    production (7f) APScheduler's POST /threads/{tid}/wake supplies the
    resume; in tests the wake is an explicit Command(resume=...).
    """
    get_logger("paper_wait").info("enter", payload={"strategy_id": state.get("strategy_id")})
    interrupt({"kind": "paper_wait", "strategy_id": state.get("strategy_id")})
    return {}


# ════════════════════════════════════════════════════════════════════════
# paper_monitor node (wraps run_paper_monitor from 7d)
# ════════════════════════════════════════════════════════════════════════


async def _build_context_from_container(state: Mapping[str, Any]) -> PaperMonitorContext:
    """Fetch live paper metrics from the container into a PaperMonitorContext.

    Default ``build_context_fn`` for :func:`paper_monitor_node`. Opens a
    short-lived Freqtrade REST client against ``freqtrade_api_url`` and
    pulls status / profit / trades / performance in one pass (snapshot
    semantics — see :mod:`orchestrator.agents.monitors`).

    ``backtest_returns`` and ``kill_switch_fired`` are read from
    ``state.artifacts`` (Stage 4 may stash backtest per-trade returns;
    Stage 8 wires the real kill-switch flag). Both degrade gracefully:
    absent backtest_returns → comparison reports not-diverged; absent
    kill flag → False.
    """
    api_url = state.get("freqtrade_api_url")
    if not api_url:
        raise RuntimeError("paper_monitor: state has no freqtrade_api_url")

    creds = FreqtradeCredentials(
        username="freqtrader",
        password=os.environ.get("PAPER_API_PASSWORD", ""),
    )
    async with FreqtradeAPI(base_url=api_url, credentials=creds) as client:
        status = await client.status()
        profit = await client.profit()
        trades_resp = await client.trades(limit=500)
        performance = await client.performance()

    trades = trades_resp.get("trades", []) if isinstance(trades_resp, dict) else []
    artifacts = state.get("artifacts") or {}
    return PaperMonitorContext(
        status=status,
        profit=profit,
        trades=trades,
        performance=performance,
        backtest_returns=artifacts.get("backtest_returns", []),
        kill_switch_fired=bool(artifacts.get("kill_switch_fired", False)),
        elapsed_days=_elapsed_paper_days(state),
    )


async def paper_monitor_node(
    state: PaperState,
    config: RunnableConfig,
    *,
    paper_monitor_fn: PaperMonitorFn | None = None,
    build_context_fn: BuildContextFn | None = None,
) -> dict[str, Any]:
    """Run the paper monitor agent and RECORD its verdict (BRD §5.5).

    This node does NOT route — it writes the verdict + the kill-relevant
    metrics into ``gate_decisions["paper_monitor"]`` and appends a vote,
    then a plain edge hands off to ``divergence_check``, which owns the
    routing. Splitting record-from-route is what lets divergence_check be
    the deterministic backstop: it reads the recorded decision + metrics
    and can OVERRIDE the LLM regardless of what the agent voted.

    Both external effects are injectable:
    - ``build_context_fn`` fetches the live snapshot (default hits the
      container; tests pass a synthetic context).
    - ``paper_monitor_fn`` invokes the agent (default = real Haiku
      run_paper_monitor; tests pass a canned-verdict stub).
    """
    get_logger("paper_monitor").info("enter", payload={"strategy_id": state.get("strategy_id")})
    build_ctx = build_context_fn or _build_context_from_container
    monitor_fn = paper_monitor_fn or run_paper_monitor

    ctx = await build_ctx(state)
    verdict = await monitor_fn(ctx)

    profit = ctx.profit if isinstance(ctx.profit, dict) else {}
    max_dd = float(profit.get("max_drawdown", 0.0) or 0.0)
    consecutive_losses = _trailing_losses(ctx.trades)

    existing = state.get("gate_decisions") or {}
    return {
        "agent_votes": [
            {
                "agent": "paper_monitor",
                "verdict": _VOTE_VERDICT[verdict.decision],
                "rationale": verdict.rationale,
                "confidence": verdict.confidence,
            }
        ],
        "gate_decisions": {
            **existing,
            "paper_monitor": {
                "decision": verdict.decision,
                "primary_observation": verdict.primary_observation,
                "rationale": verdict.rationale,
                "confidence": verdict.confidence,
                # Kill-relevant metrics divergence_check reads. These are
                # facts (not LLM judgments) so the deterministic gate may
                # use them while overriding the LLM's decision.
                "metrics": {
                    "max_drawdown": max_dd,
                    "consecutive_losses": consecutive_losses,
                },
            },
        },
    }


# ════════════════════════════════════════════════════════════════════════
# divergence_check — deterministic backstop (BRD §5.5)
# ════════════════════════════════════════════════════════════════════════


def divergence_check(
    state: PaperState,
) -> Command[Literal["paper_wait", "live_gate", "paper_teardown"]]:
    """Deterministic backstop downstream of paper_monitor (BRD §5.5).

    INDEPENDENT of the LLM verdict. Reads the recorded monitor decision +
    the kill-relevant metrics, recomputes elapsed time from the
    SPEC-locked ``paper_started_at`` clock, and applies hard overrides:

    1. **Hard kill override.** If ``max_drawdown >= KILL_SWITCH_DRAWDOWN``
       (0.12) OR ``consecutive_losses >= KILL_SWITCH_CONSECUTIVE_LOSSES``
       (10), force ``kill`` regardless of what the agent voted.
    2. **Premature-advance override.** If the (possibly already
       overridden) decision is ``advance`` but
       ``elapsed_days < MIN_PAPER_DAYS`` (30), downgrade to ``rearm``.
       The smoke proved the LLM respects this on its own; this is the
       hard guarantee that it can NEVER be bypassed.

    Routing:
      - rearm   → ``paper_wait`` (re-arm the wake cycle)
      - advance → ``live_gate`` (HITL approval to go live)
      - kill    → ``paper_teardown`` (stop container, then archive)

    Writes ``gate_decisions["divergence_check"]`` documenting the
    monitor's decision, the effective decision, elapsed days, and every
    override applied — the dashboard + audit trail show whether the
    deterministic gate intervened.
    """
    get_logger("divergence_check").info("enter", payload={"strategy_id": state.get("strategy_id")})
    gates = state.get("gate_decisions") or {}
    pm = gates.get("paper_monitor") or {}
    monitor_decision = pm.get("decision", "rearm")
    metrics = pm.get("metrics") or {}

    elapsed_days = _elapsed_paper_days(state)
    max_dd = float(metrics.get("max_drawdown", 0.0) or 0.0)
    consecutive_losses = int(metrics.get("consecutive_losses", 0) or 0)

    overrides: list[str] = []
    effective = monitor_decision

    # 1. Hard kill override — deterministic, regardless of LLM verdict.
    if max_dd >= thresholds.KILL_SWITCH_DRAWDOWN:
        if effective != "kill":
            overrides.append(
                f"hard_kill: max_drawdown={max_dd:.3f} >= "
                f"KILL_SWITCH_DRAWDOWN={thresholds.KILL_SWITCH_DRAWDOWN}"
            )
        effective = "kill"
    elif consecutive_losses >= thresholds.KILL_SWITCH_CONSECUTIVE_LOSSES:
        if effective != "kill":
            overrides.append(
                f"hard_kill: consecutive_losses={consecutive_losses} >= "
                f"KILL_SWITCH_CONSECUTIVE_LOSSES={thresholds.KILL_SWITCH_CONSECUTIVE_LOSSES}"
            )
        effective = "kill"

    # 2. Premature-advance override.
    if effective == "advance" and elapsed_days < thresholds.MIN_PAPER_DAYS:
        overrides.append(
            f"premature_advance: elapsed_days={elapsed_days:.2f} < "
            f"MIN_PAPER_DAYS={thresholds.MIN_PAPER_DAYS}; advance->rearm"
        )
        effective = "rearm"

    dc_record = {
        "monitor_decision": monitor_decision,
        "effective_decision": effective,
        "elapsed_days": elapsed_days,
        "max_drawdown": max_dd,
        "consecutive_losses": consecutive_losses,
        "overrides": overrides,
    }
    update: dict[str, Any] = {"gate_decisions": {**gates, "divergence_check": dc_record}}

    if effective == "rearm":
        return Command(goto="paper_wait", update=update)
    if effective == "advance":
        return Command(goto="live_gate", update=update)

    # kill — tear the container down first, then archive.
    reason_bits = overrides or [f"paper_monitor_kill: {pm.get('primary_observation', '')}"]
    update["failure_reason"] = "paper_divergence_kill: " + "; ".join(reason_bits)
    return Command(goto="paper_teardown", update=update)


# ════════════════════════════════════════════════════════════════════════
# live_gate — HITL interrupt (mirrors paper_gate from 6e)
# ════════════════════════════════════════════════════════════════════════


async def live_gate(
    state: PaperState,
    config: RunnableConfig,
    *,
    live_count_fn: LiveCountFn | None = None,
) -> Command[Any]:
    """Second HITL gate: approve graduates to live, reject tears down.

    Identical publish-then-interrupt / decision-on-resume structure as
    paper_gate (6e), but uses ``build_interrupt_payload(state,
    "live_gate")`` so the dashboard renders the paper_monitor rationale
    as the primary surface (the 6g renderer already handles this kind).

    Routing differs from paper_gate:
      - approve → END with ``stage="live"`` (the strategy graduates; the
        parent graph / Stage 8 live subgraph picks up from here).
      - reject  → ``paper_teardown`` (stop the dry-run container) → archive.

    **D-9 live-cap gate (Option c — BLOCK THE GATE, operator decision, BRD §10):**
    BEFORE offering HITL approval, check the live-slot capacity. If
    ``aget_portfolio_snapshot()["live"] >= MAX_CONCURRENT_LIVE_STRATEGIES`` (the
    slot is occupied), do NOT ``interrupt`` for approval and do NOT promote —
    record a ``"live_slot_occupied"`` status (surfaced in ``GET /threads`` +
    dashboard) and route back to ``paper_wait``. The existing 6-hour paper wake
    then re-runs ``paper_monitor → divergence_check → live_gate`` each wake; once
    the operator frees the slot (pauses/stops the live strategy), a later wake
    reaches here with a free slot and offers approval normally. NO auto-promotion
    — the human still approves every live entry (BRD §17 #8). NOT (a) reject/
    archive, NOT (b) auto-queue. ``live_count_fn`` is injected for hermetic tests;
    it defaults to :func:`_default_live_count`.

    Idempotency: same contract as paper_gate — publish re-fires on
    replay (harmless; Redis pubsub is non-persistent), the gate_audits
    row is written by the FastAPI endpoint not this node, and no
    provisioning happens here.
    """
    get_logger("live_gate").info("enter", payload={"strategy_id": state.get("strategy_id")})
    existing = state.get("gate_decisions") or {}

    # D-9: live-slot capacity check BEFORE presenting the approve option.
    count_fn = live_count_fn or _default_live_count
    live_count = await count_fn()
    if live_count >= thresholds.MAX_CONCURRENT_LIVE_STRATEGIES:
        get_logger("live_gate").info(
            "live_slot_occupied",
            payload={
                "strategy_id": state.get("strategy_id"),
                "live_count": live_count,
                "cap": thresholds.MAX_CONCURRENT_LIVE_STRATEGIES,
            },
        )
        return Command(
            goto="paper_wait",
            update={
                "gate_decisions": {
                    **existing,
                    "live_gate": {
                        "status": "live_slot_occupied",
                        "live_count": live_count,
                        "cap": thresholds.MAX_CONCURRENT_LIVE_STRATEGIES,
                        "by": "system",
                    },
                },
            },
        )

    payload = build_interrupt_payload(dict(state), "live_gate")
    thread_id = _thread_id_for(config, state.get("strategy_id", ""))
    await publish_gate_pending(thread_id, payload)

    decision = interrupt(payload)

    # Defensive shape validation (same as paper_gate). A malformed
    # out-of-band resume routes to teardown+archive rather than crashing.
    if not isinstance(decision, dict) or "approved" not in decision:
        return Command(
            goto="paper_teardown",
            update={
                "failure_reason": "live_gate_invalid_decision_payload",
                "gate_decisions": {
                    **existing,
                    "live": {
                        "approved": False,
                        "notes": "",
                        "by": "human",
                        "error": "invalid_decision_payload",
                    },
                },
            },
        )

    notes = str(decision.get("notes", ""))
    approved = bool(decision.get("approved", False))

    if approved:
        return Command(
            goto=END,
            update={
                "stage": "live",
                "gate_decisions": {
                    **existing,
                    "live": {"approved": True, "notes": notes, "by": "human"},
                },
            },
        )
    return Command(
        goto="paper_teardown",
        update={
            "failure_reason": f"live_gate_rejected: {notes}",
            "gate_decisions": {
                **existing,
                "live": {"approved": False, "notes": notes, "by": "human"},
            },
        },
    )


# ════════════════════════════════════════════════════════════════════════
# paper_teardown — stop container + mark registry archived
# ════════════════════════════════════════════════════════════════════════


async def paper_teardown(
    state: PaperState,
    *,
    stop_container_fn: StopContainerFn | None = None,
) -> dict[str, Any]:
    """Stop the dry-run container and mark the registry row archived.

    Reached from the kill path (divergence_check) and the reject path
    (live_gate). Calls ``stop_paper_container`` (7b) then updates
    ``strategy_registry.stage = 'archived'``. A teardown failure is
    logged but NOT raised — the thread must still archive; a leaked
    container is an operational concern surfaced via logs, not a reason
    to strand the graph. The next node (archive) stamps state.stage.
    """
    stop_fn = stop_container_fn or stop_paper_container
    strategy_id = state["strategy_id"]
    get_logger("paper_teardown").info("enter", payload={"strategy_id": strategy_id})

    try:
        await stop_fn(strategy_id)
    except Exception as exc:  # noqa: BLE001 — teardown must not strand the thread
        logger.error(
            "paper_teardown: stop_container failed strategy_id=%s exc=%s; "
            "archiving anyway (possible leaked container — check docker ps)",
            strategy_id,
            exc,
        )

    conn = await _connect_app_db()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE strategy_registry SET stage = 'archived', "
                "last_updated = now() WHERE strategy_id = %s",
                (strategy_id,),
            )
        await conn.commit()
    finally:
        await conn.close()

    # 9e (emission point A): the registry row is now committed-archived, so
    # publish thread_completed — registry-backed, post-commit (Option 1-minimal,
    # no phantom event). Self-contained + best-effort (events.publish_thread_completed
    # opens its own client and swallows Redis failures); a lost publish is
    # recovered on the next supervisor cron. paper_teardown is NOT a gate node
    # (no interrupt/replay), so an await here before the return is a plain side
    # effect alongside the stop-container + DB writes it already performs.
    await publish_thread_completed(
        strategy_id,
        {
            "strategy_id": strategy_id,
            "completed_at": datetime.now(UTC).isoformat(),
            "final_stage": "archived",
            "completion_reason": state.get("failure_reason"),
        },
    )

    return {}


# ════════════════════════════════════════════════════════════════════════
# archive — terminal sink
# ════════════════════════════════════════════════════════════════════════


def archive(state: PaperState) -> dict[str, Any]:
    """Terminal sink: stamp stage and preserve failure_reason."""
    get_logger("archive").info("enter", payload={"failure_reason": state.get("failure_reason")})
    return {
        "stage": "archived",
        "failure_reason": state.get("failure_reason") or "paper_archived_without_reason",
    }


# ════════════════════════════════════════════════════════════════════════
# Subgraph builder
# ════════════════════════════════════════════════════════════════════════


def _route_after_spawn(state: PaperState) -> Literal["schedule_wake", "archive"]:
    """Route paper_spawn's dict output: archived → archive, else continue.

    paper_spawn (7c) returns a plain dict (not a Command) so it can stay
    side-effect-described and unit-tested in isolation. This conditional
    edge translates its ``stage`` field into the spawn-failure branch.
    """
    return "archive" if state.get("stage") == "archived" else "schedule_wake"


def build_paper_subgraph(
    *,
    spawn_container_fn: SpawnContainerFn | None = None,
    paper_monitor_fn: PaperMonitorFn | None = None,
    build_context_fn: BuildContextFn | None = None,
    schedule_wake_fn: ScheduleWakeFn | None = None,
    stop_container_fn: StopContainerFn | None = None,
    live_count_fn: LiveCountFn | None = None,
    checkpointer: CheckpointSaver | None = None,
) -> CompiledStateGraph[PaperState, PaperState, PaperState, PaperState]:
    """Compile the Stage 7e paper subgraph.

    Topology (BRD §5.5)::

        START → paper_spawn ──fail──> archive ──> END
                    │ success
                    ▼
                schedule_wake → paper_wait ──(wake)──> paper_monitor
                    ▲                                       │
                    │                                       ▼
                    │                                 divergence_check
                    └──────── rearm ──────────────────────┤
                                                           ├── advance ──> live_gate
                                                           │                  ├─approve─> END (stage="live")
                                                           │                  └─reject──> paper_teardown
                                                           └── kill ─────────> paper_teardown → archive → END

    All external-effect nodes are injectable so the integration test runs
    fully stubbed (no Docker, no Postgres-spawn, no real Haiku):
    ``spawn_container_fn``, ``paper_monitor_fn``, ``build_context_fn``,
    ``schedule_wake_fn``, ``stop_container_fn``.
    """

    async def _paper_spawn(state: PaperState, config: RunnableConfig) -> dict[str, Any]:
        return await paper_spawn(
            state,  # type: ignore[arg-type]
            config,
            spawn_container_fn=spawn_container_fn,
        )

    async def _schedule_wake(state: PaperState, config: RunnableConfig) -> dict[str, Any]:
        return await schedule_wake_node(state, config, schedule_wake_fn=schedule_wake_fn)

    async def _paper_monitor(state: PaperState, config: RunnableConfig) -> dict[str, Any]:
        return await paper_monitor_node(
            state,
            config,
            paper_monitor_fn=paper_monitor_fn,
            build_context_fn=build_context_fn,
        )

    async def _paper_teardown(state: PaperState, config: RunnableConfig) -> dict[str, Any]:
        return await paper_teardown(state, stop_container_fn=stop_container_fn)

    async def _live_gate(state: PaperState, config: RunnableConfig) -> Command[Any]:
        return await live_gate(state, config, live_count_fn=live_count_fn)

    builder: StateGraph[PaperState, PaperState, PaperState, PaperState] = StateGraph(PaperState)
    builder.add_node("paper_spawn", _paper_spawn)
    builder.add_node("schedule_wake", _schedule_wake)
    builder.add_node("paper_wait", paper_wait)
    builder.add_node("paper_monitor", _paper_monitor)
    builder.add_node("divergence_check", divergence_check)
    builder.add_node("live_gate", _live_gate)
    builder.add_node("paper_teardown", _paper_teardown)
    builder.add_node("archive", archive)

    builder.add_edge(START, "paper_spawn")
    builder.add_conditional_edges("paper_spawn", _route_after_spawn, ["schedule_wake", "archive"])
    builder.add_edge("schedule_wake", "paper_wait")
    builder.add_edge("paper_wait", "paper_monitor")
    builder.add_edge("paper_monitor", "divergence_check")
    # divergence_check returns Command(goto=paper_wait|live_gate|paper_teardown)
    # live_gate returns Command(goto=END|paper_teardown)
    builder.add_edge("paper_teardown", "archive")
    builder.add_edge("archive", END)

    return builder.compile(checkpointer=checkpointer)
