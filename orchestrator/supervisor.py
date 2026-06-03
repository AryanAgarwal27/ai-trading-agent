"""Supervisor — portfolio-level read tools + registry reconciliation (Stage 9a).

BRD references:
  - §5.1: the supervisor is a ReAct agent (Sonnet 4.6) at
    ``thread_id="supervisor"`` with portfolio-level tools, running on
    APScheduler cron + event triggers.
  - §5.9: the supervisor is a reader of ``("failures", regime)``,
    ``("wins", regime)`` and ``("regime_log",)``.
  - §13 row 9 DoD: "runs nightly + on every thread completion; spawns up
    to capacity; logs decisions."

This sub-stage (9a) ships the **read-only** half only:

  - Three agent-facing tools — ``view_portfolio``, ``query_store``,
    ``get_market_regime`` — built as thin ContextVar readers, mirroring
    :mod:`orchestrator.agents.researcher`. The state they read is set by
    the Stage 9c runner before it invokes the agent; in isolation each
    tool returns a safe empty/default value (an unset store yields ``[]``,
    an unset portfolio yields a zeroed snapshot, an unset regime yields
    ``"unknown"``) — never a misleading partial answer.

  - ``sync_registry_stage(graph, conn)`` — the registry-reconciliation
    job that closes the SPEC 2026-05-27 (Stage 6f) item:
    ``strategy_registry.stage`` does NOT auto-update from LangGraph state
    (the graph is per-thread; the registry is the operator-facing
    lifecycle view, decoupled by design). The supervisor owns the
    reconciliation — when a thread transitions in the graph, the registry
    follows. ``view_portfolio`` reads the registry, so the 9c runner runs
    this FIRST so the supervisor reasons over a fresh portfolio.

Deliberately NOT in 9a: the agent itself (``create_agent`` assembly →
9c), the write tools (``spawn_strategy`` / ``retire_strategy`` →  9b),
capacity thresholds (→ 9b), and any trigger wiring (cron → 9d, event →
9e). No agent, no triggers, no graph writes here.

Seam shape (the testable boundary, for the 9c runner to wire):
  - DB reads are standalone ``async`` functions that take an explicit
    ``conn`` (caller owns the connection + transaction; they do NOT
    commit — same convention as
    :func:`orchestrator.tools.regime.insert_regime_log`). Unit tests pass
    a mocked ``psycopg.AsyncConnection``; the 9c runner opens a real one,
    runs the read, and commits/closes.
  - Agent-facing tools read three module-level ContextVars
    (``_current_store`` / ``_current_regime`` / ``_current_portfolio``).
    The 9c runner resolves each (``aget_portfolio_snapshot`` after
    ``sync_registry_stage``; ``aget_current_regime``; the open store) and
    sets the ContextVars before ``agent.ainvoke``. Per-asyncio-task
    ContextVars keep concurrent supervisor runs (event + cron overlap)
    from clobbering each other — the same reasoning researcher.py gives.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any, Literal

import psycopg
from langchain_core.tools import tool
from langgraph.store.base import BaseStore

from orchestrator.gates.thresholds import (
    MAX_CONCURRENT_STRATEGIES,
)
from orchestrator.tools.store_queries import aget_failures, aget_wins

logger = logging.getLogger(__name__)

# Stage label written to ``strategy_registry.stage`` for terminal threads.
# Reconciliation treats this as "not active" for the portfolio snapshot's
# ``active`` count (BRD §13 capacity is over in-flight, non-archived threads).
_ARCHIVED_STAGE = "archived"

# Returned by ``view_portfolio`` when the 9c runner has not (yet) resolved a
# snapshot into the ContextVar — a valid, zeroed shape so the agent always
# gets a well-formed answer rather than ``None``. In production the runner
# always sets ``_current_portfolio`` before invoking the agent.
_EMPTY_SNAPSHOT: dict[str, Any] = {"by_stage": {}, "active": 0, "live": 0}


# ─── Context-local tool inputs (set by the 9c runner) ──────────────────
# Mirrors orchestrator/agents/researcher.py: the node/runner sets these
# before invoking the agent; the state-dependent tools read them. Defaults
# are deliberately "safe empty" so a tool invoked without a runner (a unit
# test, or a misconfiguration) returns a well-formed neutral value.

_current_store: ContextVar[BaseStore | None] = ContextVar("supervisor.store", default=None)
_current_regime: ContextVar[str] = ContextVar("supervisor.regime", default="unknown")
_current_portfolio: ContextVar[dict[str, Any] | None] = ContextVar(
    "supervisor.portfolio", default=None
)

# Seam types. Both take the strategy_id and return None; the registry row the
# supervisor wrote is the handoff (spawn reads pairs/timeframe/template back).
SpawnThreadFn = Callable[[str], Awaitable[None]]
StopLiveContainerFn = Callable[[str], Awaitable[None]]


# Write-path context (set by the 9c runner alongside the read context above).
# The write @tool shells read these for the dependencies the agent does NOT
# supply — the app-DB connection, the compiled graph, and the spawn/stop
# seams. The plain ``aspawn_strategy`` / ``aretire_strategy`` impls take these
# explicitly (unit tests inject them directly); the shells bridge from the
# ContextVar the runner sets. ``None`` → the shell returns a structured
# "no_runner_context" error rather than raising into the agent loop.
_current_conn: ContextVar[psycopg.AsyncConnection | None] = ContextVar(
    "supervisor.conn", default=None
)
_current_graph: ContextVar[Any | None] = ContextVar("supervisor.graph", default=None)
_current_spawn_thread_fn: ContextVar[SpawnThreadFn | None] = ContextVar(
    "supervisor.spawn_thread_fn", default=None
)
_current_stop_live_fn: ContextVar[StopLiveContainerFn | None] = ContextVar(
    "supervisor.stop_live_fn", default=None
)

# Default spawn parameters. The supervisor seeds the registry row so the 9c
# spawn_thread_fn can read pairs/timeframe/template back to build the initial
# StrategyState (the registry row IS the spawn handoff — the seam stays
# ``spawn_thread_fn(strategy_id)``). The researcher node overwrites
# ``template`` in graph state once it chooses one; the registry seed is
# "pending" until then.
_DEFAULT_PAIRS: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT")  # SPEC §1 Q2
_DEFAULT_TIMEFRAME = "5m"
_PENDING_TEMPLATE = "pending"
_RETIRE_REASON = "retired_by_supervisor"


# ════════════════════════════════════════════════════════════════════════
# DB reads — standalone async functions (conn injected; caller owns txn)
# ════════════════════════════════════════════════════════════════════════


async def aget_portfolio_snapshot(conn: psycopg.AsyncConnection) -> dict[str, Any]:
    """Count ``strategy_registry`` rows by stage → a portfolio snapshot.

    Returns a dict with:
      - ``by_stage``: ``{stage: count}`` for every stage present.
      - ``active``: total non-archived threads (the BRD §13 "capacity"
        axis that 9b's ``spawn_strategy`` gate reads).
      - ``live``: count of ``stage="live"`` threads (the capital-bound
        axis — 9b's ``MAX_CONCURRENT_LIVE_STRATEGIES`` reads this).

    The 9c runner runs :func:`sync_registry_stage` BEFORE this so the
    counts reflect graph reality, not a stale registry. Caller owns the
    connection lifecycle; this function only reads.
    """
    async with conn.cursor() as cur:
        await cur.execute("SELECT stage, count(*) FROM strategy_registry GROUP BY stage")
        rows = await cur.fetchall()

    by_stage = {str(stage): int(count) for stage, count in rows}
    active = sum(count for stage, count in by_stage.items() if stage != _ARCHIVED_STAGE)
    live = by_stage.get("live", 0)
    return {"by_stage": by_stage, "active": active, "live": live}


async def aget_current_regime(conn: psycopg.AsyncConnection) -> str:
    """Return the most recent ``regime_log`` label, or ``"unknown"`` (BRD §5.9).

    The regime APScheduler job (Stage 7f, :func:`orchestrator.scheduler.
    _fire_regime_job`) writes ``regime_log`` rows hourly; the supervisor
    reads the latest as the anchor regime for ``query_store`` scoping and
    for its own spawn reasoning. An empty table (fresh install, no market
    data yet) yields ``"unknown"`` — the same sentinel researcher.py uses
    — never an error. Caller owns the connection lifecycle.
    """
    async with conn.cursor() as cur:
        await cur.execute("SELECT regime FROM regime_log ORDER BY at DESC LIMIT 1")
        row = await cur.fetchone()
    return str(row[0]) if row else "unknown"


async def sync_registry_stage(graph: Any, conn: psycopg.AsyncConnection) -> dict[str, Any]:
    """Reconcile ``strategy_registry.stage`` with each thread's graph state.

    Closes the SPEC 2026-05-27 (Stage 6f) item: the registry stage does
    not auto-update from LangGraph state, so it drifts (it reflects the
    last manually-set / supervisor-written value). This reads each
    registry row's thread state via ``graph.aget_state`` and writes back
    any stage that has moved.

    Behaviour:
      - A thread whose graph ``state.stage`` differs from its registry
        ``stage`` is UPDATEd (``stage`` + ``last_updated = now()``).
      - A thread already in sync is left untouched (no write).
      - A thread with no checkpoint, an ``aget_state`` error, or a state
        carrying no ``stage`` is SKIPPED (logged) — a registry row can
        legitimately exist before its graph thread has run (9b's
        ``spawn_strategy`` inserts the row, then kicks the graph), and
        clobbering its seed stage to nothing would be wrong.

    Connection contract: the caller owns ``conn`` and its transaction —
    this function executes the UPDATEs but does **not** commit (same
    convention as :func:`orchestrator.tools.regime.insert_regime_log`).
    The 9c runner wraps a ``sync → snapshot`` read in one transaction and
    commits once.

    Returns a summary ``{"checked", "updated", "skipped"}`` for the
    decision log + tests: ``updated`` is a list of
    ``{"strategy_id", "from", "to"}`` transitions; ``skipped`` is a list
    of strategy_ids left as-is.
    """
    async with conn.cursor() as cur:
        await cur.execute("SELECT strategy_id, thread_id, stage FROM strategy_registry")
        rows = await cur.fetchall()

    updated: list[dict[str, str]] = []
    skipped: list[str] = []

    for strategy_id, thread_id, registry_stage in rows:
        config = {"configurable": {"thread_id": thread_id}}
        try:
            snapshot = await graph.aget_state(config)
        except Exception as exc:  # noqa: BLE001 — one bad thread must not abort the sweep
            logger.warning(
                "sync_registry_stage: aget_state failed thread_id=%s err=%s; skipping",
                thread_id,
                exc,
            )
            skipped.append(str(strategy_id))
            continue

        values = getattr(snapshot, "values", None) or {}
        graph_stage = values.get("stage")
        if not graph_stage:
            # Registry row with no checkpoint / no stage in state — a
            # freshly-seeded row whose graph thread has not run yet, or a
            # thread that never wrote a stage. Leave the registry as-is.
            skipped.append(str(strategy_id))
            continue

        if graph_stage == registry_stage:
            continue

        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE strategy_registry SET stage = %s, last_updated = now() "
                "WHERE strategy_id = %s",
                (graph_stage, strategy_id),
            )
        updated.append(
            {"strategy_id": str(strategy_id), "from": str(registry_stage), "to": str(graph_stage)}
        )
        logger.info("sync_registry_stage: %s %s -> %s", strategy_id, registry_stage, graph_stage)

    return {"checked": len(rows), "updated": updated, "skipped": skipped}


# ════════════════════════════════════════════════════════════════════════
# Tool implementations (plain functions — unit-tested directly)
# ════════════════════════════════════════════════════════════════════════
# The @tool wrappers below are thin ContextVar-reading shells over these.
# Splitting the logic out keeps the unit tests free of langchain tool-
# invocation / cross-thread-ContextVar concerns — they call the _impl
# directly after setting the ContextVar, exactly as the runner will.


def _view_portfolio_impl() -> dict[str, Any]:
    snapshot = _current_portfolio.get()
    # Return a COPY of the zeroed default so a caller mutating the result
    # can never corrupt the module-level constant.
    return snapshot if snapshot is not None else dict(_EMPTY_SNAPSHOT)


def _get_market_regime_impl() -> str:
    return _current_regime.get()


async def _query_store_impl(
    category: Literal["failures", "wins"],
    regime: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    store = _current_store.get()
    if store is None:
        return []
    target_regime = regime or _current_regime.get()
    if category == "failures":
        return await aget_failures(store, target_regime, limit=limit)
    return await aget_wins(store, target_regime, limit=limit)


# ════════════════════════════════════════════════════════════════════════
# Agent-facing tools (ContextVar readers — wired into create_agent in 9c)
# ════════════════════════════════════════════════════════════════════════


@tool
def view_portfolio() -> dict[str, Any]:
    """Return the current portfolio snapshot: thread counts by lifecycle stage.

    Returns a dict with ``by_stage`` (``{stage: count}``), ``active``
    (total non-archived threads — your capacity headroom is the gap
    between this and the configured maximum), and ``live`` (threads
    currently trading real capital). Call this before deciding whether to
    spawn: spawning beyond capacity is refused. An all-zero snapshot means
    no strategies exist yet (a fresh install), not an error.
    """
    return _view_portfolio_impl()


@tool
async def query_store(
    category: Literal["failures", "wins"],
    regime: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return past ``failures`` or ``wins`` from the long-term Store (BRD §5.9).

    Args:
        category: ``"failures"`` for strategies that lost money / failed a
            gate, ``"wins"`` for strategies that completed a live cycle
            profitably.
        regime: Regime label to scope the lookup (e.g. ``"low_vol_up"``).
            Defaults to the current market regime (see
            ``get_market_regime``) when omitted.
        limit: Max records to return. Default 10.

    Returns:
        A list of dicts carrying each archived strategy's hypothesis,
        params, failure_reason (failures only) and live_metrics_summary
        (wins only). An empty list means nothing is recorded for this
        regime yet — common on a fresh install; not an error.
    """
    return await _query_store_impl(category, regime, limit)


@tool
def get_market_regime() -> str:
    """Return the current market regime label (BRD §5.7, §5.9).

    Composite ``{vol}_{trend}`` label (e.g. ``"high_vol_up"``) from the
    most recent ``regime_log`` row, or ``"unknown"`` when no
    classification is available yet. Use it to scope ``query_store`` and
    to anchor spawn decisions to the regime that will actually be live.
    """
    return _get_market_regime_impl()


# The read-only tool surface. The Stage 9b write tools (spawn_strategy /
# retire_strategy) and the 9c agent assembly extend this list; exporting
# it now gives those sub-stages a single import point and locks the 9a
# surface for the tests.
READ_TOOLS = [view_portfolio, query_store, get_market_regime]


# ════════════════════════════════════════════════════════════════════════
# Write tools — portfolio mutations (Stage 9b)
# ════════════════════════════════════════════════════════════════════════
# Plain async impls (unit-tested directly with injected conn/graph/seams);
# the @tool shells below read the write-path ContextVars and delegate.
# Caller-owns-commit throughout — the 9c runner batches reconciliation +
# these writes in one transaction (matches sync_registry_stage's contract).


async def aspawn_strategy(
    spawn_thread_fn: SpawnThreadFn,
    conn: psycopg.AsyncConnection,
    *,
    name: str | None = None,
    template: str | None = None,
    pairs: list[str] | None = None,
    timeframe: str | None = None,
) -> dict[str, Any]:
    """Mint a strategy, capacity-gate it, seed the registry, kick the graph.

    Pipeline:
      1. Read :func:`aget_portfolio_snapshot` for the current ``active`` count.
      2. **Capacity gate (resource axis):** if
         ``active >= MAX_CONCURRENT_STRATEGIES``, REFUSE — return
         ``{"spawned": False, "reason": "capacity_exceeded", ...}`` WITHOUT
         inserting a registry row or calling ``spawn_thread_fn``. This is the
         money-relevant guard: a spawn that slips past it consumes a
         container slot the host can't afford.
      3. Else: mint a UUID ``strategy_id`` (``thread_id = strategy_<id>``),
         insert a ``stage="research"`` registry row carrying the spawn
         parameters (so ``spawn_thread_fn`` can read them back to build the
         initial StrategyState — the registry row is the handoff), and call
         ``spawn_thread_fn(strategy_id)`` exactly once.

    The ``MAX_CONCURRENT_LIVE_STRATEGIES`` (capital axis) cap is deliberately
    NOT enforced here — it binds at the paper→live transition, not at research
    spawn (see the threshold docstring). Spawning is gated only by the total
    cap so the pipeline keeps researching/papering while one strategy is live.

    Caller owns the connection + transaction; this function does NOT commit.
    """
    snapshot = await aget_portfolio_snapshot(conn)
    active = snapshot["active"]
    if active >= MAX_CONCURRENT_STRATEGIES:
        logger.info(
            "spawn refused: active=%d >= MAX_CONCURRENT_STRATEGIES=%d",
            active,
            MAX_CONCURRENT_STRATEGIES,
        )
        return {
            "spawned": False,
            "reason": "capacity_exceeded",
            "active": active,
            "limit": MAX_CONCURRENT_STRATEGIES,
        }

    strategy_id = uuid.uuid4().hex
    thread_id = f"strategy_{strategy_id}"
    resolved_name = name or f"strategy_{strategy_id[:8]}"
    resolved_template = template or _PENDING_TEMPLATE
    resolved_pairs = pairs if pairs is not None else list(_DEFAULT_PAIRS)
    resolved_timeframe = timeframe or _DEFAULT_TIMEFRAME

    async with conn.cursor() as cur:
        await cur.execute(
            """
            INSERT INTO strategy_registry
              (strategy_id, thread_id, name, template, stage, pairs,
               timeframe, started_at, last_updated)
            VALUES (%s, %s, %s, %s, 'research', %s, %s, now(), now())
            ON CONFLICT (strategy_id) DO NOTHING
            """,
            (
                strategy_id,
                thread_id,
                resolved_name,
                resolved_template,
                json.dumps(resolved_pairs),
                resolved_timeframe,
            ),
        )

    # Kick the per-strategy graph. The seam runs the graph from START until
    # its first interrupt (production impl lands with the 9c runner); tests
    # inject a recording stub. Called exactly once, AFTER the registry row
    # exists so spawn_thread_fn can read the spawn parameters back.
    await spawn_thread_fn(strategy_id)

    logger.info("spawned strategy_id=%s (active was %d)", strategy_id, active)
    return {
        "spawned": True,
        "strategy_id": strategy_id,
        "thread_id": thread_id,
        "stage": "research",
    }


async def aretire_strategy(
    graph: Any,
    conn: psycopg.AsyncConnection,
    strategy_id: str,
    *,
    stop_live_container_fn: StopLiveContainerFn | None = None,
) -> dict[str, Any]:
    """Archive a thread: halt its live container (if any), mark it archived.

    This does NOT route through the graph — it performs the archive EFFECTS
    directly, then marks the graph state so a later ``aget_state`` reads the
    thread as archived:

      1. ``aget_state`` the thread. No checkpoint (empty values) → graceful
         no-op ``{"retired": False, "reason": "unknown_strategy"}`` (mirrors
         the kill-subscription Fork-2 guard — never mint a phantom checkpoint).
      2. If the thread is ``stage="live"``, call ``stop_live_container_fn``
         (best-effort: a stop failure is logged, not raised, so it cannot
         strand the archive). Default seam =
         :func:`orchestrator.tools.freqtrade_lifecycle.stop_live_container`.
      3. ``aupdate_state`` → ``stage="archived"`` + ``failure_reason`` so the
         thread reads as terminal and the parent graph routes it to END on
         any subsequent touch.
      4. UPDATE the ``strategy_registry`` row → ``stage="archived"``.

    Caller owns the connection + transaction; this function does NOT commit.

    NOTE (surfaced in the 9b handoff): the supervisor should retire threads
    that are PARKED or idle. ``aupdate_state`` on a thread actively executing
    in a background spawn task (mid-research / mid-Send-fanout in validation)
    would race that task; retiring such a transient thread is unsupported in
    v1. The aupdate_state also clears the parent-visible interrupt surface for
    a nested-subgraph park (the 8h finding) — desired here (the thread is no
    longer resumable) but it leaves any APScheduler wake job for the thread
    dangling (it 409s harmlessly on its next fire); wake-job cleanup is a
    teardown detail deferred to the 9f live-wake work.
    """
    thread_id = f"strategy_{strategy_id}"
    config = {"configurable": {"thread_id": thread_id}}

    snapshot = await graph.aget_state(config)
    values = getattr(snapshot, "values", None) or {}
    if not values:
        logger.warning("retire: thread %s has no checkpoint; no-op (unknown strategy)", thread_id)
        return {"retired": False, "reason": "unknown_strategy", "strategy_id": strategy_id}

    previous_stage = values.get("stage")

    if previous_stage == "live":
        stop_fn = stop_live_container_fn
        if stop_fn is None:
            # Lazy import keeps supervisor.py import-light; the real stop is
            # only needed when actually retiring a live thread.
            from orchestrator.tools.freqtrade_lifecycle import stop_live_container

            stop_fn = stop_live_container
        try:
            await stop_fn(strategy_id)
        except Exception as exc:  # noqa: BLE001 — a stop failure must not strand the archive
            logger.error(
                "retire: stop_live_container failed strategy_id=%s exc=%s; "
                "archiving anyway (operator reviews orphaned container)",
                strategy_id,
                exc,
            )

    await graph.aupdate_state(config, {"stage": "archived", "failure_reason": _RETIRE_REASON})

    async with conn.cursor() as cur:
        await cur.execute(
            "UPDATE strategy_registry SET stage = 'archived', "
            "failure_reason = %s, last_updated = now() WHERE strategy_id = %s",
            (_RETIRE_REASON, strategy_id),
        )

    logger.info("retired strategy_id=%s (was %s)", strategy_id, previous_stage)
    return {"retired": True, "strategy_id": strategy_id, "previous_stage": previous_stage}


# ─── Agent-facing write tools (ContextVar readers — wired in 9c) ───────


@tool
async def spawn_strategy(
    name: str | None = None,
    template: str | None = None,
    pairs: list[str] | None = None,
    timeframe: str | None = None,
) -> dict[str, Any]:
    """Spawn a new strategy research thread, if portfolio capacity allows.

    Kicks off a fresh strategy lifecycle at the ``research`` stage. All args
    are optional hints — leave them unset to let the researcher choose the
    template and use the default v1 pair universe.

    Returns ``{"spawned": True, "strategy_id", "thread_id", "stage"}`` on
    success, or ``{"spawned": False, "reason": "capacity_exceeded", "active",
    "limit"}`` when the portfolio is already at the concurrent-strategy cap —
    call ``view_portfolio`` first to check headroom. Spawning beyond capacity
    is refused, not queued.
    """
    conn = _current_conn.get()
    spawn_fn = _current_spawn_thread_fn.get()
    if conn is None or spawn_fn is None:
        return {"spawned": False, "reason": "no_runner_context"}
    return await aspawn_strategy(
        spawn_fn, conn, name=name, template=template, pairs=pairs, timeframe=timeframe
    )


@tool
async def retire_strategy(strategy_id: str) -> dict[str, Any]:
    """Retire (archive) an active strategy thread by its ``strategy_id``.

    Halts the strategy's live Freqtrade container if it is trading, then
    marks the thread and its registry row ``archived``. Use this to free a
    capacity slot — e.g. to make room under the concurrent-strategy cap, or
    to retire a strategy that has stopped earning. Returns ``{"retired":
    True, "previous_stage"}`` on success, or ``{"retired": False, "reason":
    "unknown_strategy"}`` if no such thread exists.
    """
    conn = _current_conn.get()
    graph = _current_graph.get()
    stop_fn = _current_stop_live_fn.get()
    if conn is None or graph is None:
        return {"retired": False, "reason": "no_runner_context"}
    return await aretire_strategy(graph, conn, strategy_id, stop_live_container_fn=stop_fn)


# Combined surfaces for the 9c agent assembly. WRITE_TOOLS = the two mutations;
# SUPERVISOR_TOOLS = the full toolset create_agent receives.
WRITE_TOOLS = [spawn_strategy, retire_strategy]
SUPERVISOR_TOOLS = READ_TOOLS + WRITE_TOOLS
