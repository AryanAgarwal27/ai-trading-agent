"""Supervisor — portfolio-level orchestrator (Stage 9a/9b/9c).

BRD references:
  - §5.1: the supervisor is a ReAct agent (Sonnet 4.6) at
    ``thread_id="supervisor"`` with portfolio-level tools, running on
    APScheduler cron + event triggers.
  - §2: system overview — the supervisor spawns strategy threads.
  - §13 row 9 DoD: "runs nightly + on every thread completion; spawns up
    to capacity; logs decisions."
  - §5.9: the supervisor reads ``("failures", regime)`` / ``("wins", regime)``.

Layered across the 9-series sub-stages, all in this module:

  - **9a — read surface + reconciliation.** Three ContextVar-reader tools
    (``view_portfolio`` / ``query_store`` / ``get_market_regime``) mirroring
    :mod:`orchestrator.agents.researcher`, plus ``sync_registry_stage`` (closes
    the SPEC 2026-05-27 Stage-6f item: the registry stage does not auto-update
    from LangGraph state, so the supervisor reconciles it).

  - **9b — write impls + capacity gate.** ``aspawn_strategy`` (gated on
    ``MAX_CONCURRENT_STRATEGIES``) and ``aretire_strategy`` — plain async
    functions, caller-owns-commit.

  - **9c — agent + runner.** ``list_strategies`` (the per-thread read tool the
    agent needs to choose a retire target), ``SupervisorDecision`` structured
    output, ``build_supervisor_agent`` (Sonnet 4.6 + SUPERVISOR_TOOLS +
    structured output, mirroring researcher.py), and ``run_supervisor`` — one
    end-to-end invocation: reconcile → resolve read context → invoke agent →
    execute the decision's write actions → log the decision to telemetry →
    single batch commit → post-commit spawn drain.

Architecture (Arch 2, operator fork sign-off): the agent is bound with the
READ tools only + ``response_format=SupervisorDecision``. It reasons and emits
INTENT; it executes nothing. The runner is the single source of execution
truth — it calls the plain ``aspawn_strategy`` / ``aretire_strategy`` impls
from ``decision.actions``. This mirrors researcher.py exactly (read tools +
structured output, node executes) and keeps a deterministic, single-transaction
audit trail. (The 9b agent-facing write ``@tool`` shells were removed under
this architecture — they had no caller; the plain impls are what the runner
and 9b's tests use.)

Supervisor persistence: ``build_supervisor_agent`` wires NO checkpointer, so
the agent is stateless-per-run — each invocation reasons fresh from the
snapshot/regime/strategies in the kickoff message, and the durable decision
record is the telemetry row, not an LLM conversation. ``thread_id="supervisor"``
is passed in the invoke config as the conceptual thread id only (harmless
without a checkpointer, and ready to become persistent ONLY if a future
sub-stage deliberately reverses this v1 design — see DEFERRED.md D-8).
Consequence: D-8's unbounded-context concern is dormant by design — it cannot
occur while the supervisor stays stateless.

D-10 (registry↔graph-state mirror contract) is CLOSED in Stage 10f:
``sync_registry_stage`` now mirrors the full declared :data:`MIRRORED_FIELDS`
set (``stage`` + ``template``) per the :func:`_mirror_value` overwrite-vs-skip
rule, not stage alone. ``alist_strategies`` still exposes ``last_transition_at``
via ``last_updated`` (the registry has no dedicated per-transition column — a
Stage 11 candidate, out of D-10's mirror-contract scope).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import psycopg
from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from pydantic import BaseModel, ConfigDict, Field

from orchestrator.agents.generator import SHIPPED_TEMPLATES
from orchestrator.gates.thresholds import MAX_CONCURRENT_STRATEGIES
from orchestrator.observability.events import (
    _connect_app_db,
    publish_thread_completed,
    record_telemetry,
)
from orchestrator.observability.log import run_context
from orchestrator.observability.metrics import SUPERVISOR_RUNS, set_strategies_by_stage
from orchestrator.observability.tracing import trace_config
from orchestrator.tools.store_queries import aget_failures, aget_wins

logger = logging.getLogger(__name__)

# Stage label written to ``strategy_registry.stage`` for terminal threads.
# Reconciliation treats this as "not active" for the portfolio snapshot's
# ``active`` count (BRD §13 capacity is over in-flight, non-archived threads).
_ARCHIVED_STAGE = "archived"

# Returned by ``view_portfolio`` when the runner has not (yet) resolved a
# snapshot into the ContextVar — a valid, zeroed shape so the agent always gets
# a well-formed answer rather than ``None``.
_EMPTY_SNAPSHOT: dict[str, Any] = {"by_stage": {}, "active": 0, "live": 0}

# The supervisor's conceptual thread id (BRD §5.1) + telemetry source tag.
_SUPERVISOR_THREAD_ID = "supervisor"
_AUDIT_SOURCE = "supervisor_decision"

# Default spawn parameters. ``aspawn_strategy`` seeds the registry row so the
# producer (``_default_spawn_thread_fn``) can read pairs/timeframe/template back
# to build the initial StrategyState (the registry row IS the spawn handoff).
# The researcher node overwrites ``template`` in graph state once it chooses
# one; the registry seed is "pending" until ``sync_registry_stage`` mirrors the
# chosen value back per the mirror contract below (D-10, closed Stage 10f).
_DEFAULT_PAIRS: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT")  # SPEC §1 Q2
# The pair universe (SPEC §1 Q2) — the spawn-vocabulary whitelist for pairs,
# derived from _DEFAULT_PAIRS so the two never drift. A spawn naming any pair
# outside this set has no cached OHLCV / Binance whitelist entry and is rejected
# at the spawn boundary (D-16). Single source of truth = _DEFAULT_PAIRS above;
# the template whitelist is generator.SHIPPED_TEMPLATES (BRD §8.1).
_PAIR_UNIVERSE: frozenset[str] = frozenset(_DEFAULT_PAIRS)
_DEFAULT_TIMEFRAME = "5m"
_PENDING_TEMPLATE = "pending"
_RETIRE_REASON = "retired_by_supervisor"


# ─── Context-local read inputs (set by run_supervisor before agent.ainvoke) ──
# Mirrors orchestrator/agents/researcher.py: the runner resolves each value
# (via the conn-injected aget_*/alist_* functions) and sets these; the
# agent-facing read tools are thin readers. Defaults are "safe empty" so a tool
# invoked without a runner returns a well-formed neutral value. Set inside
# run_supervisor's try/finally and reset before return (runner-scoped — NOT
# inherited by the post-commit spawn tasks).

_current_store: ContextVar[BaseStore | None] = ContextVar("supervisor.store", default=None)
_current_regime: ContextVar[str] = ContextVar("supervisor.regime", default="unknown")
_current_portfolio: ContextVar[dict[str, Any] | None] = ContextVar(
    "supervisor.portfolio", default=None
)
_current_strategies: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "supervisor.strategies", default=None
)


# Seam types. spawn/stop take the strategy_id (the registry row is the handoff);
# the audit writer takes the caller's conn + the metrics payload (no commit).
SpawnThreadFn = Callable[[str], Awaitable[None]]
StopLiveContainerFn = Callable[[str], Awaitable[None]]
AuditWriterFn = Callable[[psycopg.AsyncConnection, dict[str, Any]], Awaitable[None]]
# 9e: (strategy_id, payload) → publish a thread_completed event. Injected so the
# runner's post-commit retire emission is hermetic in unit tests; defaults to
# events.publish_thread_completed (self-contained, best-effort).
CompletionPublisherFn = Callable[[str, dict[str, Any]], Awaitable[None]]
# 9f (D-6): (strategy_id) → cancel that strategy's recurring live-wake job.
# aretire_strategy calls it ONLY when the retired thread was live (only live
# threads have a live_wake job). Default no-op so retire works without a
# scheduler (tests / standalone); production injects make_unschedule_live_wake_fn.
UnscheduleWakeFn = Callable[[str], Awaitable[None]]


async def _noop_unschedule_wake(strategy_id: str) -> None:
    return None


# ════════════════════════════════════════════════════════════════════════
# DB reads — standalone async functions (conn injected; caller owns txn)
# ════════════════════════════════════════════════════════════════════════


async def aget_portfolio_snapshot(conn: psycopg.AsyncConnection) -> dict[str, Any]:
    """Count ``strategy_registry`` rows by stage → a portfolio snapshot.

    Returns ``by_stage`` (``{stage: count}``), ``active`` (total non-archived
    threads — the BRD §13 capacity axis ``aspawn_strategy`` gates on), and
    ``live`` (``stage="live"`` count — the capital axis). The runner runs
    :func:`sync_registry_stage` BEFORE this so counts reflect graph reality.
    Caller owns the connection; this only reads.
    """
    async with conn.cursor() as cur:
        await cur.execute("SELECT stage, count(*) FROM strategy_registry GROUP BY stage")
        rows = await cur.fetchall()

    by_stage = {str(stage): int(count) for stage, count in rows}
    active = sum(count for stage, count in by_stage.items() if stage != _ARCHIVED_STAGE)
    live = by_stage.get("live", 0)
    # Stage 10e (BRD §14): publish the per-stage gauge from the just-computed
    # registry counts — this is the count-by-stage site, so the Prometheus
    # gauge and the supervisor's own snapshot are one source of truth.
    set_strategies_by_stage(by_stage)
    return {"by_stage": by_stage, "active": active, "live": live}


async def aget_current_regime(conn: psycopg.AsyncConnection) -> str:
    """Return the most recent ``regime_log`` label, or ``"unknown"`` (BRD §5.9).

    The regime APScheduler job (Stage 7f) writes ``regime_log`` rows hourly;
    the supervisor reads the latest as the anchor regime for ``query_store``
    scoping and spawn reasoning. Empty table → ``"unknown"`` (same sentinel
    researcher.py uses). Caller owns the connection.
    """
    async with conn.cursor() as cur:
        await cur.execute("SELECT regime FROM regime_log ORDER BY at DESC LIMIT 1")
        row = await cur.fetchone()
    return str(row[0]) if row else "unknown"


async def alist_strategies(conn: psycopg.AsyncConnection) -> list[dict[str, Any]]:
    """Per-thread view of every NON-ARCHIVED strategy (BRD §5.1).

    Returns one dict per active thread with:
      - ``strategy_id`` (pass to retire), ``stage``, ``name``.
      - ``age_days``: days since ``started_at`` (overall age).
      - ``last_transition_at``: the registry's ``last_updated`` ISO timestamp —
        the best-available proxy for "when the thread last transitioned"
        (``last_updated`` is touched on stage transitions + a few non-transition
        writes; the registry has no dedicated per-transition column). The agent
        uses it as a stall signal. A dedicated transition column is a 9h/D-10
        candidate.

    Ordered oldest-transition-first so the agent sees the most-stalled threads
    at the top. Caller owns the connection; this only reads.
    """
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT strategy_id, stage, name, "
            "EXTRACT(EPOCH FROM (now() - started_at)) / 86400.0 AS age_days, "
            "last_updated "
            "FROM strategy_registry WHERE stage != 'archived' "
            "ORDER BY last_updated ASC"
        )
        rows = await cur.fetchall()

    out: list[dict[str, Any]] = []
    for strategy_id, stage, name, age_days, last_updated in rows:
        out.append(
            {
                "strategy_id": str(strategy_id),
                "stage": str(stage),
                "name": str(name),
                "age_days": round(float(age_days), 1) if age_days is not None else None,
                "last_transition_at": last_updated.isoformat()
                if last_updated is not None
                else None,
            }
        )
    return out


# ════════════════════════════════════════════════════════════════════════
# Registry ↔ graph-state MIRROR CONTRACT (D-10, declared ONCE here)
# ════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class MirrorField:
    """One ``strategy_registry`` column that MIRRORS graph state.

    A field is mirrored IFF the GRAPH is its source of truth — the registry
    value is a seed placeholder the graph later derives. ``seed_sentinels`` are
    the placeholder values that mean "not yet derived"; the overwrite-vs-skip
    rule (:func:`_mirror_value`) overwrites the registry only once graph state
    holds a real (truthy, non-sentinel) value that differs.
    """

    name: str
    seed_sentinels: frozenset[str]


# WHICH registry columns mirror graph state, and the overwrite-vs-skip rule —
# D-10's deliverable: declared in ONE place for ALL derived fields, applied
# generically below (not a stage-only / one-off-template patch).
#
#   IN (graph is the source of truth):
#     - ``stage``    — lifecycle position; every transition happens in the graph.
#                      No seed sentinel: every Stage value is real.
#     - ``template`` — seeded ``"pending"`` at research-spawn; the researcher
#                      chooses the real template in graph state later
#                      (researcher.py). ``"pending"`` is the seed sentinel.
#
#   OUT (the REGISTRY is the source of truth — NOT mirrored):
#     - ``pairs``, ``timeframe``, ``name`` — spawn INPUTS. ``aspawn_strategy``
#       sets them from the supervisor/operator's spawn decision; they are seeded
#       into the registry AND passed into the graph as read-only inputs. NO graph
#       node writes them back into ``StrategyState``, so mirroring them would be
#       a no-op at best and could clobber the operator's spawn record if a node
#       ever rewrote them. (``freqtrade_api_url`` / ``freqtrade_userdir`` /
#       ``live_started_at`` are written to the registry directly by the
#       paper/live spawn nodes — they are container facts, not graph-derived
#       mirror fields, so they are out of scope here too.)
#
# OVERWRITE-VS-SKIP RULE (D-10's open question — declared for ALL fields):
#   graph state WINS for a mirrored field once it holds a REAL value — overwrite
#   the registry iff the graph value is truthy, NOT a seed sentinel, and differs
#   from the registry value. Otherwise SKIP — never overwrite a real registry
#   value with an empty/seed graph value. This generalizes the stage-only
#   behavior that shipped in 9a.
MIRRORED_FIELDS: tuple[MirrorField, ...] = (
    MirrorField(name="stage", seed_sentinels=frozenset()),
    MirrorField(name="template", seed_sentinels=frozenset({_PENDING_TEMPLATE})),
)

# SELECT column list for the mirror sweep — identity columns + every mirrored
# field, in declaration order (so row[2:] aligns with MIRRORED_FIELDS).
_MIRROR_SELECT_COLUMNS = ", ".join(f.name for f in MIRRORED_FIELDS)


def _mirror_value(field: MirrorField, registry_value: Any, graph_value: Any) -> bool:
    """The declared overwrite-vs-skip rule. ``True`` ⇒ overwrite the registry.

    Graph state wins once it holds a real value: truthy, not a seed sentinel,
    and different from the current registry value. A missing/empty graph value
    or one still equal to a seed sentinel keeps the registry as-is.
    """
    if not graph_value:  # None / "" / empty — the graph has not produced it yet
        return False
    if graph_value in field.seed_sentinels:  # still the seed placeholder
        return False
    return bool(graph_value != registry_value)


async def sync_registry_stage(graph: Any, conn: psycopg.AsyncConnection) -> dict[str, Any]:
    """Reconcile the registry's graph-derived columns with each thread's state.

    Closes the SPEC 2026-05-27 (Stage 6f) item AND D-10: the registry does not
    auto-update from LangGraph state, so its graph-derived fields drift. Reads
    each registry row's thread state via ``graph.aget_state`` and writes back
    every mirrored field that moved, per the :data:`MIRRORED_FIELDS` contract +
    the :func:`_mirror_value` overwrite-vs-skip rule above.

    Behaviour:
      - For each row, every mirrored field (``stage``, ``template``) whose graph
        value WINS (truthy, non-seed, differs) is written; all winners for a row
        go in ONE UPDATE (``<col> = %s, …`` + ``last_updated = now()``).
      - A row already in sync for every mirrored field is left untouched.
      - A thread with no checkpoint, an ``aget_state`` error, or an empty state
        (a freshly-seeded row whose graph thread has not run yet) is SKIPPED — it
        keeps every seed value.

    Name kept as ``sync_registry_stage`` for call-site stability (the runner +
    cron + endpoint reference it); it now syncs the full mirror set, not stage
    alone. Caller owns ``conn`` + the transaction; this does NOT commit (the
    runner batches sync + writes + telemetry into one commit). Returns
    ``{"checked", "updated", "skipped"}`` where ``updated`` is one change record
    per field: ``{"strategy_id", "field", "from", "to"}``.
    """
    async with conn.cursor() as cur:
        await cur.execute(
            f"SELECT strategy_id, thread_id, {_MIRROR_SELECT_COLUMNS} FROM strategy_registry"
        )
        rows = await cur.fetchall()

    updated: list[dict[str, str]] = []
    skipped: list[str] = []

    for row in rows:
        strategy_id, thread_id = row[0], row[1]
        registry_values = dict(zip((f.name for f in MIRRORED_FIELDS), row[2:], strict=True))
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
        if not values:
            # Graph thread has not run yet — keep every seed value.
            skipped.append(str(strategy_id))
            continue

        # Collect the mirrored fields whose graph value wins, in declaration
        # order (so the UPDATE + change records are deterministic).
        changes: list[tuple[MirrorField, Any]] = []
        for mfield in MIRRORED_FIELDS:
            graph_value = values.get(mfield.name)
            if _mirror_value(mfield, registry_values[mfield.name], graph_value):
                changes.append((mfield, graph_value))

        if not changes:
            continue

        set_clause = ", ".join(f"{mfield.name} = %s" for mfield, _ in changes)
        params = tuple(new_value for _, new_value in changes) + (strategy_id,)
        async with conn.cursor() as cur:
            await cur.execute(
                f"UPDATE strategy_registry SET {set_clause}, last_updated = now() "
                "WHERE strategy_id = %s",
                params,
            )
        for mfield, new_value in changes:
            old_value = registry_values[mfield.name]
            updated.append(
                {
                    "strategy_id": str(strategy_id),
                    "field": mfield.name,
                    "from": str(old_value),
                    "to": str(new_value),
                }
            )
            logger.info(
                "sync_registry_stage: %s %s %s -> %s",
                strategy_id,
                mfield.name,
                old_value,
                new_value,
            )

    return {"checked": len(rows), "updated": updated, "skipped": skipped}


# ════════════════════════════════════════════════════════════════════════
# Read-tool implementations (plain — unit-tested directly via ContextVars)
# ════════════════════════════════════════════════════════════════════════


def _view_portfolio_impl() -> dict[str, Any]:
    snapshot = _current_portfolio.get()
    # Return a COPY of the zeroed default so a mutating caller can't corrupt it.
    return snapshot if snapshot is not None else dict(_EMPTY_SNAPSHOT)


def _get_market_regime_impl() -> str:
    return _current_regime.get()


def _list_strategies_impl() -> list[dict[str, Any]]:
    strategies = _current_strategies.get()
    return list(strategies) if strategies is not None else []


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
# Agent-facing read tools (ContextVar readers — bound into create_agent)
# ════════════════════════════════════════════════════════════════════════


@tool
def view_portfolio() -> dict[str, Any]:
    """Return the current portfolio snapshot: thread counts by lifecycle stage.

    Returns a dict with ``by_stage`` (``{stage: count}``), ``active`` (total
    non-archived threads — your capacity headroom is the gap between this and
    the configured maximum), and ``live`` (threads trading real capital). Call
    this before deciding whether to spawn: spawning beyond capacity is refused.
    An all-zero snapshot means no strategies exist yet, not an error.
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
        category: ``"failures"`` for strategies that lost money / failed a gate,
            ``"wins"`` for strategies that completed a live cycle profitably.
        regime: Regime label to scope the lookup (e.g. ``"low_vol_up"``).
            Defaults to the current market regime when omitted.
        limit: Max records to return. Default 10.

    Returns:
        A list of dicts carrying each archived strategy's hypothesis, params,
        failure_reason (failures only) and live_metrics_summary (wins only).
        Empty list means nothing recorded for this regime yet — not an error.
    """
    return await _query_store_impl(category, regime, limit)


@tool
def get_market_regime() -> str:
    """Return the current market regime label (BRD §5.7, §5.9).

    Composite ``{vol}_{trend}`` label (e.g. ``"high_vol_up"``) from the most
    recent ``regime_log`` row, or ``"unknown"`` when none is available. Use it
    to scope ``query_store`` and anchor spawn decisions to the live regime.
    """
    return _get_market_regime_impl()


@tool
def list_strategies() -> list[dict[str, Any]]:
    """List the active (non-archived) strategy threads in the portfolio.

    Returns one row per non-archived thread: ``strategy_id`` (pass to
    ``retire``), ``stage`` (research / validation / paper_gate / paper /
    live_gate / live), ``name``, ``age_days`` (overall age since the strategy
    was created), and ``last_transition_at`` (when its registry row last
    changed — a proxy for how long it has sat in its current stage). Rows are
    ordered most-stalled-first. Use this to pick a retire target — e.g. a
    thread parked at a HITL gate for many days. Empty list = no active threads.
    """
    return _list_strategies_impl()


# The 9a-frozen read surface (test_read_tools_surface pins exactly these three).
READ_TOOLS = [view_portfolio, query_store, get_market_regime]

# The full toolset the agent receives (Arch 2: READ tools only — the agent
# emits a SupervisorDecision; the runner executes the writes). list_strategies
# is appended here rather than into READ_TOOLS to keep the 9a surface frozen.
SUPERVISOR_TOOLS = READ_TOOLS + [list_strategies]


# ════════════════════════════════════════════════════════════════════════
# Write impls — portfolio mutations (Stage 9b; called by the runner, Arch 2)
# ════════════════════════════════════════════════════════════════════════
# Plain async functions (unit-tested directly with injected conn/graph/seams).
# Caller-owns-commit throughout — run_supervisor batches reconciliation + these
# writes + the telemetry row into one transaction.


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

    1. Read :func:`aget_portfolio_snapshot` for the current ``active`` count.
    2. **Capacity gate (resource axis):** if ``active >=
       MAX_CONCURRENT_STRATEGIES``, REFUSE — return ``{"spawned": False,
       "reason": "capacity_exceeded", ...}`` WITHOUT inserting a row or calling
       ``spawn_thread_fn``. The money-relevant guard.
    3. Else: mint a UUID ``strategy_id`` (``thread_id = strategy_<id>``), insert
       a ``stage="research"`` row carrying the spawn parameters (the handoff for
       the producer), and call ``spawn_thread_fn(strategy_id)`` exactly once.

    ``MAX_CONCURRENT_LIVE_STRATEGIES`` is deliberately NOT enforced here — it
    binds at the paper→live transition (D-9), not at research spawn.

    **Spawn-vocabulary gate (D-16):** BEFORE the capacity read, reject a spawn
    whose ``template`` is not a shipped template
    (:data:`generator.SHIPPED_TEMPLATES`, BRD §8.1) or whose ``pairs`` are not
    all inside the SPEC §1 Q2 universe (:data:`_PAIR_UNIVERSE`) — return
    ``{"spawned": False, "reason": "unknown_template" | "pairs_outside_universe",
    ...}`` WITHOUT inserting a row or calling ``spawn_thread_fn`` (the same
    no-write contract as the capacity refusal). We REJECT the whole action rather
    than clamp the offending field: a hallucinated template means the agent
    misunderstood its option set, and silently half-executing (dropping the
    template to ``pending``, filtering bad pairs) would seed a registry row that
    contradicts the action's own rationale — the operator audit would then
    misrepresent what was decided, and the slot would be spent on something the
    agent did not intend. ``template=None`` / ``pairs=None`` are the normal
    "researcher chooses the template / default universe applies" case and pass
    the gate. Defense-in-depth: ``_SUPERVISOR_PROMPT`` also enumerates the valid
    vocabulary, but this gate — not the prompt — is the guarantee.

    Caller owns the connection + transaction; this does NOT commit.
    """
    if template is not None and template not in SHIPPED_TEMPLATES:
        logger.warning(
            "spawn refused: template=%r not in shipped templates %s "
            "(D-16 vocabulary gate) — no registry row written",
            template,
            sorted(SHIPPED_TEMPLATES),
        )
        return {
            "spawned": False,
            "reason": "unknown_template",
            "template": template,
            "allowed_templates": sorted(SHIPPED_TEMPLATES),
        }
    if pairs is not None:
        invalid_pairs = [p for p in pairs if p not in _PAIR_UNIVERSE]
        if invalid_pairs:
            logger.warning(
                "spawn refused: pairs %s outside the SPEC §1 Q2 universe %s "
                "(D-16 vocabulary gate) — no registry row written",
                invalid_pairs,
                sorted(_PAIR_UNIVERSE),
            )
            return {
                "spawned": False,
                "reason": "pairs_outside_universe",
                "invalid_pairs": invalid_pairs,
                "allowed_pairs": sorted(_PAIR_UNIVERSE),
            }

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

    # Kick the per-strategy graph. Under the runner this seam ENQUEUES the
    # strategy_id; the runner drains the queue AFTER commit so the kicked graph
    # run sees the committed registry row (Option A). Called once, after the
    # row exists.
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
    unschedule_wake_fn: UnscheduleWakeFn | None = None,
) -> dict[str, Any]:
    """Archive a thread: halt its live container (if any), mark it archived.

    Performs the archive EFFECTS directly (no graph re-entry, no transitional
    state):
      1. ``aget_state`` — empty values → ``{"retired": False, "reason":
         "unknown_strategy"}`` (Fork-2 guard; never mint a phantom checkpoint).
      2. If ``stage="live"``, best-effort ``stop_live_container_fn`` (a failure
         is logged, not raised — it can't strand the archive). Default seam =
         :func:`orchestrator.tools.freqtrade_lifecycle.stop_live_container`.
      3. ``aupdate_state`` → ``stage="archived"`` + ``failure_reason``.
      4. UPDATE the registry row → ``stage="archived"``.

    Caller owns the connection + transaction; this does NOT commit.

    Race contract (v1): retire PARKED/idle threads only. ``aupdate_state`` on a
    thread actively executing in a post-commit spawn task (mid-research /
    mid-fanout) would race it; locking is a Stage-11 concern. It also clears the
    parent interrupt surface for a nested park (8h finding) — desired here — but
    leaves any APScheduler wake job dangling (409s harmlessly; cleanup → 9f).
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

    # 9f (D-6): only a LIVE thread has a recurring live-wake job — cancel it so
    # it doesn't fire forever and 409. Best-effort: a cleanup failure logs but
    # never strands the retire (the registry archive already happened; a missed
    # cancel only costs harmless 409s). Caller-owns-commit is unaffected — this
    # touches the scheduler, not the conn.
    if previous_stage == "live":
        try:
            await (unschedule_wake_fn or _noop_unschedule_wake)(strategy_id)
        except Exception as exc:  # noqa: BLE001 — cleanup must not strand the retire
            logger.error("retire: unschedule_wake failed strategy_id=%s exc=%s", strategy_id, exc)

    logger.info("retired strategy_id=%s (was %s)", strategy_id, previous_stage)
    return {"retired": True, "strategy_id": strategy_id, "previous_stage": previous_stage}


# ════════════════════════════════════════════════════════════════════════
# Structured output — the supervisor's decision (Stage 9c, Piece 1)
# ════════════════════════════════════════════════════════════════════════


class SupervisorAction(BaseModel):
    """One action the supervisor commits to. Field names align 1:1 with
    ``aspawn_strategy`` / ``aretire_strategy`` so the runner maps trivially."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["spawn", "retire", "no_op"] = Field(
        description="spawn a new research thread, retire an existing one, or no_op.",
    )
    # spawn hints (all optional — researcher picks the template; defaults cover
    # pairs/timeframe).
    name: str | None = Field(default=None, description="spawn: optional human label.")
    template: str | None = Field(
        default=None,
        description="spawn: optional template hint; omit to let the researcher choose.",
    )
    pairs: list[str] | None = Field(
        default=None, description="spawn: optional pair list; omit for the v1 default universe."
    )
    timeframe: str | None = Field(
        default=None, description="spawn: optional timeframe; omit for 5m."
    )
    # retire target.
    strategy_id: str | None = Field(
        default=None, description="retire: the strategy_id (from list_strategies) to archive."
    )
    rationale: str = Field(
        description="REQUIRED for every action — the operator audit reads this.",
    )


class SupervisorDecision(BaseModel):
    """The supervisor's full decision for one run — the agent's only output."""

    model_config = ConfigDict(extra="forbid")

    actions: list[SupervisorAction] = Field(
        description="Ordered actions to take. An empty list (or a single no_op) is valid.",
    )
    overall_rationale: str = Field(description="Why this set of actions, as a whole.")
    confidence: float = Field(ge=0.0, le=1.0, description="Self-assessed confidence, 0.0–1.0.")


# ════════════════════════════════════════════════════════════════════════
# System prompt + agent build (Stage 9c, Piece 2)
# ════════════════════════════════════════════════════════════════════════


_SUPERVISOR_PROMPT = """\
You are the Supervisor — the portfolio-level orchestrator of an autonomous
crypto-trading agent (BRD §5.1, §2). You do not trade and you do not research
strategies yourself. Your only powers are to SPAWN new strategy research
threads and RETIRE existing ones, keeping the portfolio inside its capacity
budget and biased toward what has worked.

You run nightly and on events (a strategy thread completing). Each run you
survey the portfolio and emit a structured decision — a list of actions, each
with a rationale. You do not produce free text; you commit to actions.

Protocol, every run:

1. Call view_portfolio() FIRST. It returns counts by stage plus `active`
   (total non-archived threads) and `live`. Your hard constraint is
   MAX_CONCURRENT_STRATEGIES, which is 4: propose a spawn ONLY while
   `active` < 4. If `active` >= 4, do not spawn — retire to make room, or
   no_op.

2. Call get_market_regime(), then query_store("failures", regime=<current>)
   and query_store("wins", regime=<current>) (BRD §5.9). Past failures in this
   regime are expensive lessons: do NOT spawn into a regime with dense recent
   failures unless you can state specifically how this proposal differs from
   what failed. Prefer building on a recorded win when one exists.

3. Call list_strategies() to see active threads (id, stage, name, age_days,
   last_transition_at). Consider RETIRING a strategy stalled at paper_gate or
   live_pause_review for more than ~14 days with no operator action — a
   long-stalled HITL gate is a proxy for operator disinterest and it holds a
   capacity slot. GUIDELINE, not a rule: a strategy mid-paper-run (stage
   "paper", not stalled at a gate) is doing its job — do not retire it for
   being slow.

4. Emit a SupervisorDecision: actions (spawn / retire / no_op) each with a
   rationale, plus overall_rationale and confidence (0.0–1.0). An empty action
   list (or a single no_op) is a legitimate and common outcome — most nights
   the right move is to let existing threads run. Spawning for activity's sake
   burns the capacity budget and the operator's review attention.

Rules:
- Never propose more spawns than the remaining headroom (4 − active).
- Every action needs a concrete rationale. "Spawn a strategy" is not one;
  "spawn a mean-reversion candidate — wins in mid_vol_flat favor it and 2 slots
  are free" is.
- You cannot start/stop/approve trades or move a strategy through its gates —
  those are the per-strategy graph's and the operator's jobs. Spawn and retire
  are your only levers.
- A past failure in this regime does not permanently ban the underlying idea
  (template + pairs). It does require the rationale to name how this proposal
  differs from what failed — different parameters, different pair, different
  regime expectation. "Same idea, hoping for a different outcome" is the
  definition of what the failure namespace was built to prevent.
"""

# D-16: enumerate the REAL spawn vocabulary in the prompt so the agent stops
# inventing templates/pairs (the 10d real-trace proposed templates `stat_arb` /
# `momentum` and pair `AVAX/USDT`, none of which exist). This is best-effort
# guidance that REDUCES how often the gate fires — the load-bearing guarantee is
# aspawn_strategy's spawn-vocabulary gate, not this text. Built from the canonical
# sets (SHIPPED_TEMPLATES / _PAIR_UNIVERSE) so the prompt can never drift from the
# validator.
_SUPERVISOR_PROMPT += (
    "\nSpawn vocabulary (HARD constraint — a spawn naming anything outside these "
    "lists is REJECTED at the spawn boundary before any row is written, wasting the "
    "run):\n"
    f"- template: OMIT it (preferred — the researcher chooses), or name EXACTLY one "
    f"of: {', '.join(sorted(SHIPPED_TEMPLATES))}. Do NOT invent templates.\n"
    f"- pairs: omit for the default universe, or use a SUBSET of: "
    f"{', '.join(sorted(_PAIR_UNIVERSE))}. No other pairs exist.\n"
)


def build_supervisor_agent() -> Any:
    """Build the Sonnet 4.6 ReAct agent (BRD §4, §5.1).

    Mirrors :mod:`orchestrator.agents.researcher` and
    :mod:`orchestrator.agents.coordinator`: lazy import + construction (so this
    module imports without ``ANTHROPIC_API_KEY``), Sonnet 4.6 with
    ``timeout=60.0`` + ``stop=None``, READ tools bound (Arch 2 — the agent emits
    intent; the runner executes writes), and ``response_format=SupervisorDecision``
    forcing structured output.

    No checkpointer is wired — the agent is stateless-per-run (see the module
    docstring on supervisor persistence / D-8). The runner passes
    ``thread_id="supervisor"`` in the invoke config as the conceptual thread id.
    """
    from langchain.agents import create_agent
    from langchain_anthropic import ChatAnthropic

    model = ChatAnthropic(model="claude-sonnet-4-6", timeout=60.0, stop=None)
    return create_agent(
        model=model,
        tools=SUPERVISOR_TOOLS,
        system_prompt=_SUPERVISOR_PROMPT,
        response_format=SupervisorDecision,
    )


# ════════════════════════════════════════════════════════════════════════
# Runner (Stage 9c, Piece 3) + production spawn producer (Piece 4)
# ════════════════════════════════════════════════════════════════════════

# Hold references to fire-and-forget spawn tasks so they aren't GC'd mid-run.
_BACKGROUND_SPAWN_TASKS: set[asyncio.Task[None]] = set()


async def _default_audit_writer(conn: psycopg.AsyncConnection, metrics: dict[str, Any]) -> None:
    """Default ``audit_writer_fn`` — one ``telemetry`` row, no commit (Fork C).

    ``source="supervisor_decision"``, ``strategy_id=None`` (portfolio-level),
    payload in ``metrics``. Joins the runner's single batch commit.
    """
    await record_telemetry(
        conn, strategy_id=None, stage=None, metrics=metrics, source=_AUDIT_SOURCE
    )


async def _default_spawn_thread_fn(graph: Any, strategy_id: str) -> None:
    """Production spawn producer (Piece 4) — kick the per-strategy graph.

    Reads the (now-committed) registry row for the spawn parameters, builds the
    initial StrategyState, and starts ``graph.astream`` from START as a
    FIRE-AND-FORGET background task. Returns immediately after scheduling — the
    supervisor must never block on per-strategy execution time (the parent
    graph's first superstep is the entire research subgraph, which is LLM-heavy,
    so we do NOT await the first yield). Input validation (registry row present)
    happens synchronously before backgrounding. A runtime failure inside the run
    is logged and surfaces as an archived thread with a failure_reason — it does
    not roll back the already-committed spawn decision.
    """
    conn = await _connect_app_db()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT name, template, pairs, timeframe FROM strategy_registry "
                "WHERE strategy_id = %s",
                (strategy_id,),
            )
            row = await cur.fetchone()
    finally:
        await conn.close()

    if row is None:
        logger.error("spawn producer: no registry row for %s; cannot kick graph", strategy_id)
        return

    name, template, pairs, timeframe = row
    now_iso = datetime.now(UTC).isoformat()
    initial_state: dict[str, Any] = {
        "strategy_id": strategy_id,
        "name": name,
        "template": template,
        "pairs": pairs,
        "timeframe": timeframe,
        "stage": "research",
        "started_at": now_iso,
        "last_updated": now_iso,
        "artifacts": {},
    }
    config = {"configurable": {"thread_id": f"strategy_{strategy_id}"}}

    async def _drive() -> None:
        try:
            # Stage 10c: the initial spawn invoke is one GRAPH EXECUTION — bind a
            # fresh execution-scoped run_id for the whole research→… run so every
            # node it drives logs under the same id. run_id is NOT carried on
            # initial_state (it must never be checkpointed — BRD §5.7); it lives
            # only in structlog contextvars for this task's scope.
            # Stage 10d: thread that SAME run_id into trace_config so the spawn's
            # LangSmith trace shares one id with its structlog lines. Entry stage
            # is "research" (initial_state["stage"]) — the lifecycle phase the
            # execution begins in.
            with run_context(
                strategy_id=strategy_id, thread_id=f"strategy_{strategy_id}"
            ) as run_id:
                traced_config = trace_config(
                    config,
                    strategy_id=strategy_id,
                    thread_id=f"strategy_{strategy_id}",
                    run_id=run_id,
                    stage="research",
                )
                async for _ in graph.astream(initial_state, config=traced_config):
                    pass
        except Exception as exc:  # noqa: BLE001 — background run; surfaces via logs + archived thread
            logger.error("spawn producer: graph run failed strategy_id=%s exc=%s", strategy_id, exc)

    task = asyncio.create_task(_drive())
    _BACKGROUND_SPAWN_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_SPAWN_TASKS.discard)


async def _invoke_supervisor_agent(
    agent: Any,
    trigger: str,
    timestamp: str,
) -> SupervisorDecision | None:
    """Invoke the agent; return its SupervisorDecision, or None on flake.

    The kickoff message is deliberately MINIMAL — trigger + timestamp only, NO
    embedded snapshot/regime/strategies (9c smoke finding). The runner has
    already resolved those into the read-tool ContextVars, so the agent fetches
    them by CALLING view_portfolio / get_market_regime / query_store /
    list_strategies as its protocol mandates. Embedding the data in the message
    let the agent skip those tool calls (the data was already in front of it),
    leaving the telemetry + LangSmith trace an unfaithful record of what it
    actually queried. Minimal kickoff → the trace reflects real reasoning.

    None signals an agent flake (raised, or returned no/invalid structured
    output). The caller falls back to a no_op decision — flakes must not crash
    the run.
    """
    from langchain_core.messages import HumanMessage

    kickoff = (
        f"Begin supervisor protocol. Trigger: {trigger}. Run started at {timestamp}.\n\n"
        "Survey the portfolio, query the regime + Store, and emit a "
        "SupervisorDecision per your protocol."
    )
    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=kickoff)]},
            config={"configurable": {"thread_id": _SUPERVISOR_THREAD_ID}},
        )
    except Exception as exc:  # noqa: BLE001 — agent flake must not crash run_supervisor
        logger.error("supervisor agent invocation failed: %s", exc)
        return None

    resp = result.get("structured_response") if isinstance(result, dict) else None
    if not isinstance(resp, SupervisorDecision):
        logger.error("supervisor agent returned no/invalid structured_response (%r)", type(resp))
        return None
    return resp


async def _execute_action(
    action: SupervisorAction,
    graph: Any,
    conn: psycopg.AsyncConnection,
    enqueue_fn: SpawnThreadFn,
    unschedule_wake_fn: UnscheduleWakeFn | None = None,
) -> dict[str, Any]:
    """Execute one decision action via the plain write impls (Arch 2).

    Returns a per-action result dict (including a refusal like
    ``capacity_exceeded``) so the telemetry row records what actually happened,
    not just what the agent intended.
    """
    if action.action == "spawn":
        result = await aspawn_strategy(
            enqueue_fn,
            conn,
            name=action.name,
            template=action.template,
            pairs=action.pairs,
            timeframe=action.timeframe,
        )
        return {"action": "spawn", "result": result}
    if action.action == "retire":
        if not action.strategy_id:
            return {
                "action": "retire",
                "result": {"retired": False, "reason": "missing_strategy_id"},
            }
        result = await aretire_strategy(
            graph, conn, action.strategy_id, unschedule_wake_fn=unschedule_wake_fn
        )
        return {"action": "retire", "result": result}
    return {"action": "no_op", "result": {"rationale": action.rationale}}


# D-11: ONE fixed advisory-lock key serializing EVERY run_supervisor invocation
# (cron + event + manual). The 9d cron (id="supervisor_cron") and 9e event
# (id="supervisor_event_oneshot") runs are DISTINCT APScheduler job ids, so
# max_instances=1 (which is PER-job-id) does NOT serialize them against each
# other — two runs could read the capacity gate concurrently and momentarily
# overshoot MAX_CONCURRENT_STRATEGIES by 1. pg_advisory_xact_lock is
# TRANSACTION-scoped: acquired before sync_registry_stage, it is held for the
# whole snapshot→reason→spawn→commit sequence and AUTO-RELEASES when this run's
# transaction ends (commit / rollback / connection close) — crash-safe, no leaked
# lock. A concurrent run BLOCKS on acquire, then reads capacity only AFTER the
# holder commits, closing the read-then-write window.
#
# Choices (stated per the D-11 brief): xact-scoped over SESSION-scoped
# (pg_advisory_lock) because a session lock survives a crashed run and would
# strand the next invocation — xact auto-release on conn close is the crash-safe
# property we want. Blocking pg_advisory_xact_lock over pg_try_* / skip because a
# WAIT serializes without DROPPING the second run: it still reasons, on fresh
# post-commit state, rather than no-op'ing the event/cron tick.
_SUPERVISOR_ADVISORY_LOCK_KEY: int = 0x4149_545F_5355_5056  # "AIT_SUPV" ascii; fixed/arbitrary


async def _acquire_supervisor_lock(conn: psycopg.AsyncConnection) -> None:
    """Acquire the transaction-scoped advisory lock serializing run_supervisor.

    Blocks until granted; auto-released when ``conn``'s transaction ends (commit /
    rollback / close). The single serialization primitive for D-11 (and the
    capacity read-then-write window it closes).
    """
    async with conn.cursor() as cur:
        await cur.execute("SELECT pg_advisory_xact_lock(%s)", (_SUPERVISOR_ADVISORY_LOCK_KEY,))


async def run_supervisor(
    graph: Any,
    store: BaseStore,
    conn: psycopg.AsyncConnection,
    *,
    trigger: Literal["cron", "event", "manual"],
    agent: Any | None = None,
    spawn_thread_fn: SpawnThreadFn | None = None,
    audit_writer_fn: AuditWriterFn | None = None,
    completion_publisher_fn: CompletionPublisherFn | None = None,
    unschedule_wake_fn: UnscheduleWakeFn | None = None,
    dry_run: bool = False,
) -> SupervisorDecision:
    """One end-to-end supervisor invocation (BRD §5.1, §13 row 9).

    Sequence: ``sync_registry_stage`` → snapshot/regime/strategies reads → set
    read-tool ContextVars → invoke the agent at ``thread_id="supervisor"`` →
    execute ``decision.actions`` via the plain write impls → write one telemetry
    decision row → SINGLE batch commit → post-commit spawn drain.

    Transaction: every DB write (sync UPDATEs, spawn INSERTs, retire UPDATEs,
    the telemetry row) runs on ``conn`` and commits together at the end — the
    caller-owns-commit contract 9a/9b were built for. On any exception the whole
    batch rolls back and the error re-raises; a single supervisor failure cannot
    leave the registry half-mutated.

    Resilience: an agent that flakes (raises, or returns malformed structured
    output) does NOT crash the run — it falls back to a no_op decision flagged
    ``flake=True``, still writes telemetry, and still commits.

    ContextVar scope: the read-tool ContextVars (``_current_store``,
    ``_current_regime``, ``_current_portfolio``, ``_current_strategies``) are set
    inside a try/finally and reset before this function returns. They are NOT
    inherited by the asyncio.Tasks spawned post-commit to kick per-strategy graph
    runs — those tasks read everything they need from the committed registry row
    + their own arguments. Cross-task ContextVar inheritance is not supported and
    not relied on.

    Spawn ordering (Option A): during the action loop ``spawn_thread_fn`` merely
    ENQUEUES the strategy_id; after the commit the runner drains the queue,
    calling the real producer — so the kicked graph run reads a COMMITTED
    registry row on its own connection (BRD §5.3), never a dirty one.

    Dry run: ``dry_run=True`` executes the full READ + REASONING path (sync →
    snapshot → regime → strategies → ContextVars → agent → decision extraction,
    including the flake fallback) but SKIPS all writes — no action execution
    (no spawn/retire), no telemetry row, no spawn drain — and ``conn.rollback()``
    replaces the commit so any read-path mutations (e.g. a ``sync_registry_stage``
    UPDATE) are unwound. Returns the SupervisorDecision unchanged. This is the
    path the smoke probe (``scripts/smoke_supervisor.py``) drives so it tests the
    REAL production reasoning sequence — the agent calls the read tools as its
    protocol mandates — without mutating the DB.

    Thread-completion emission (Stage 9e, point C — Option 1-minimal): for each
    action that ACTUALLY archived a registry row this run (a ``retire`` whose
    impl returned ``retired=True``), a ``thread_completed`` event is published
    AFTER the batch commit, in the post-commit block alongside the spawn drain —
    so every event has a backing committed-archived row (no phantom). Spawns,
    no_ops, capacity-refused retires, and ``sync_registry_stage`` transitions
    (point D) deliberately do NOT emit; the nightly cron is the backstop for
    those. Best-effort per emission: a publish failure is logged and swallowed
    (the committed archive is the source of truth), never rolling back.

    Seams: ``agent`` (default ``build_supervisor_agent()``), ``spawn_thread_fn``
    (default the real producer bound to ``graph``), ``audit_writer_fn`` (default
    the telemetry writer), ``completion_publisher_fn`` (default
    :func:`orchestrator.observability.events.publish_thread_completed`),
    ``unschedule_wake_fn`` (9f — cancels a retired LIVE thread's recurring
    live-wake job; default no-op, production injects
    :func:`orchestrator.scheduler.make_unschedule_live_wake_fn`). Tests inject
    them to stay hermetic.
    """
    # Stage 10e (BRD §14): count this run by trigger (cron / event / manual) at
    # the real run site — once per invocation, before any work.
    SUPERVISOR_RUNS.labels(trigger=trigger).inc()

    audit = audit_writer_fn or _default_audit_writer
    publish_completion = completion_publisher_fn or publish_thread_completed
    spawn_queue: list[str] = []
    retired_strategy_ids: list[str] = []

    async def _enqueue(sid: str) -> None:
        spawn_queue.append(sid)

    async def _producer(sid: str) -> None:
        if spawn_thread_fn is not None:
            await spawn_thread_fn(sid)
        else:
            await _default_spawn_thread_fn(graph, sid)

    try:
        # D-11: serialize all run_supervisor invocations on ONE xact-scoped
        # advisory lock BEFORE any read — closes the capacity read-then-write race
        # across the distinct cron/event/manual job ids. Held until the commit /
        # rollback below; auto-released on conn close (crash-safe).
        await _acquire_supervisor_lock(conn)
        await sync_registry_stage(graph, conn)
        snapshot = await aget_portfolio_snapshot(conn)
        regime = await aget_current_regime(conn)
        strategies = await alist_strategies(conn)

        tok_store = _current_store.set(store)
        tok_regime = _current_regime.set(regime)
        tok_portfolio = _current_portfolio.set(snapshot)
        tok_strategies = _current_strategies.set(strategies)
        try:
            run_agent = agent or build_supervisor_agent()
            decision = await _invoke_supervisor_agent(
                run_agent, trigger, datetime.now(UTC).isoformat()
            )
        finally:
            _current_store.reset(tok_store)
            _current_regime.reset(tok_regime)
            _current_portfolio.reset(tok_portfolio)
            _current_strategies.reset(tok_strategies)

        flake = decision is None
        if decision is None:
            decision = SupervisorDecision(
                actions=[],
                overall_rationale="agent output unavailable/malformed; defaulting to no_op",
                confidence=0.0,
            )

        if dry_run:
            # Read + reasoning path only — no writes. Roll back any read-path
            # mutation (a sync_registry_stage UPDATE) and return the decision.
            # The post-commit spawn drain below is skipped (queue is empty).
            await conn.rollback()
            logger.info("run_supervisor dry_run: decision computed; writes skipped, rolled back")
            return decision

        action_results = [
            await _execute_action(action, graph, conn, _enqueue, unschedule_wake_fn)
            for action in decision.actions
        ]

        # 9e: collect the strategy_ids that ACTUALLY archived this run (retire
        # impl returned retired=True) — only those transitioned a registry row,
        # so only those get a post-commit thread_completed event (no phantom).
        retired_strategy_ids = [
            r["result"]["strategy_id"]
            for r in action_results
            if r.get("action") == "retire" and r.get("result", {}).get("retired") is True
        ]

        metrics = {
            "trigger": trigger,
            "flake": flake,
            "decision": decision.model_dump(),
            "action_results": action_results,
        }
        await audit(conn, metrics)

        await conn.commit()
    except Exception:
        await conn.rollback()
        logger.exception("run_supervisor failed; rolled back the batch")
        raise

    # Post-commit spawn drain (Option A) — committed rows are now visible to the
    # per-strategy graph kicks on their own connections.
    for sid in spawn_queue:
        try:
            await _producer(sid)
        except Exception as exc:  # noqa: BLE001 — one bad kick must not abort the drain
            logger.error("run_supervisor: spawn kick failed strategy_id=%s exc=%s", sid, exc)

    # Post-commit thread_completed emission (9e point C) — the retire UPDATEs are
    # now committed-archived, so publish AFTER the commit (registry-backed, no
    # phantom). Best-effort per emission: publish_thread_completed already
    # swallows Redis failures, and this try/except is the belt-and-braces drain
    # guard (one bad publish must not abort the rest); a missed publish is
    # recovered on the next cron. Skipped on the dry_run path (returns earlier).
    for sid in retired_strategy_ids:
        try:
            await publish_completion(
                sid,
                {
                    "strategy_id": sid,
                    "completed_at": datetime.now(UTC).isoformat(),
                    "final_stage": _ARCHIVED_STAGE,
                    "completion_reason": _RETIRE_REASON,
                },
            )
        except Exception as exc:  # noqa: BLE001 — one bad publish must not abort the drain
            logger.error(
                "run_supervisor: thread_completed publish failed strategy_id=%s exc=%s", sid, exc
            )

    return decision
