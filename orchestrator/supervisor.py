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

import logging
from contextvars import ContextVar
from typing import Any, Literal

import psycopg
from langchain_core.tools import tool
from langgraph.store.base import BaseStore

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
