"""Per-strategy parent graph (BRD §5.2).

Composes the lifecycle subgraphs in sequence. Stage 7g wired research →
validation → paper; Stage 8g wires the live subgraph in after paper
graduates, completing the BRD §5.2 four-subgraph topology.

Topology (post-8g)::

    START
      │
      ▼
    research_subgraph ──archived──> END
      │ pass
      ▼
    validation_subgraph ──archived──> END
      │ stage="paper" (paper_gate approved)
      ▼
    paper_subgraph ──archived──> END
      │ stage="live" (live_gate approved)
      ▼
    live_subgraph ──> END   (exits at stage="archived" — live_pause reject
                             or coordinator fail; live_wait/live_pause park
                             on interrupts in between)

Each subgraph is a nested compiled graph added as a parent node. A
dynamic ``interrupt()`` inside a nested subgraph (paper_gate in
validation; paper_wait + live_gate in paper; live_wait + live_pause in
live) DOES surface to the parent's ``aget_state(config).tasks[*].interrupts``
— verified empirically in Stage 7g — so the FastAPI ``GET /threads``
endpoint sees pending interrupts through the real parent graph. The parent
task NAME for a nested interrupt is the SUBGRAPH NODE NAME (e.g.
``validation_subgraph`` / ``live_subgraph``), NOT the inner gate name, so
the resume endpoints (``/approve``, ``/wake``) identify the gate by the
interrupt payload's ``"kind"`` field rather than the task name (see
orchestrator/main.py).

Inter-subgraph routing keys off ``state["stage"]``:
- research sets ``stage="archived"`` on its failure paths; a pass leaves
  stage unchanged → route to validation.
- validation's paper_gate approve sets ``stage="paper"`` → route to
  paper; any archived path → END.
- paper exits at ``stage="live"`` (live_gate approved) → route to live;
  ``stage="archived"`` (kill / reject) → END.
- live exits at ``stage="archived"`` (live_pause reject / coordinator
  fail) → END; a continue re-arms live_wait inside the subgraph (it does
  not exit on continue).

Every subgraph is injectable (``research_subgraph`` /
``validation_subgraph`` / ``paper_subgraph`` / ``live_subgraph``) so
integration tests can compose with lightweight stub subgraphs and exercise
the parent routing + interrupt surfacing without Docker, real LLMs, or
cached market data. Production passes none of these and the real subgraphs
are built from the leaf-effect seams (``worker_fn``, ``risk_analyst_fn``,
``paper_monitor_fn``, ``spawn_container_fn``, ``schedule_wake_fn``,
``spawn_live_container_fn``, ``build_snapshot_fn`` …).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.postgres.aio import AsyncPostgresStore

from orchestrator.agents.coordinator import ArbitrateFn, GateAuditWriterFn, MergeFn
from orchestrator.security.secrets import SecretProvider
from orchestrator.state import BacktestResult, StrategyState
from orchestrator.subgraphs.live import (
    BuildSnapshotFn,
    LiveStartedBumpFn,
    PerformanceReviewFn,
    RationaleFn,
    RegistryWriterFn,
    SpawnLiveContainerFn,
    StartTradingFn,
    StopTradingFn,
    build_live_subgraph,
)
from orchestrator.subgraphs.live import (
    StopContainerFn as LiveStopContainerFn,
)
from orchestrator.subgraphs.paper import (
    BuildContextFn,
    PaperMonitorFn,
    ScheduleWakeFn,
    SpawnContainerFn,
    StopContainerFn,
    build_paper_subgraph,
)
from orchestrator.subgraphs.research import build_research_subgraph
from orchestrator.subgraphs.validation import (
    BacktestWorkerFn,
    RiskAnalystFn,
    build_validation_subgraph,
)
from orchestrator.tools.backtest_runner import run_backtest


async def _default_backtest_worker_fn(payload: dict[str, Any]) -> BacktestResult:
    """Production backtest worker — wraps ``run_backtest`` on a Send payload.

    Mirrors the closure ``tests/integration/test_validation_subgraph.py``
    uses. The Send payload is ``{**state, "_param_set": ps, "_fold": fold}``
    (see ``validation.plan_backtests``).

    NOTE (Stage 7g flag): this reads ``payload["strategy_path"]`` and
    ``ps["id"]`` — the field names the validation subgraph + its tests
    use. The research subgraph's output → validation input mapping
    (does research emit ``strategy_path`` / ``param_sets[*].id`` /
    ``folds``?) is NOT yet verified against a real research run; the
    composition test stubs the worker, so this default is unexercised
    until a real end-to-end run. Tracked as a research→validation
    field-handoff verification item.
    """
    ps = payload["_param_set"]
    fold = payload["_fold"]
    return await run_backtest(
        Path(payload["strategy_path"]),
        pairs=payload["pairs"],
        timeframe=payload["timeframe"],
        timerange=fold["timerange"],
        fold_id=fold["fold_id"],
        param_set_id=ps["id"],
    )


def _route_after_research(state: StrategyState) -> Literal["validation", "end"]:
    """research archived → END; otherwise advance to validation."""
    return "end" if state.get("stage") == "archived" else "validation"


def _route_after_validation(state: StrategyState) -> Literal["paper", "end"]:
    """validation paper_gate-approve (stage=paper) → paper; else → END."""
    return "paper" if state.get("stage") == "paper" else "end"


def _route_after_paper(state: StrategyState) -> Literal["live", "end"]:
    """paper live_gate-approve (stage=live) → live; archived/anything else → END.

    The paper subgraph (BRD §5.5) exits with exactly two terminal stages:
    ``stage="live"`` (live_gate approved → graduate) or ``stage="archived"``
    (kill / reject). Route the former into the live subgraph; the latter is
    terminal.
    """
    return "live" if state.get("stage") == "live" else "end"


def build_per_strategy_graph(
    saver: AsyncPostgresSaver | Any,
    store: AsyncPostgresStore | Any,
    *,
    research_subgraph: CompiledStateGraph[Any, Any, Any, Any] | None = None,
    validation_subgraph: CompiledStateGraph[Any, Any, Any, Any] | None = None,
    paper_subgraph: CompiledStateGraph[Any, Any, Any, Any] | None = None,
    live_subgraph: CompiledStateGraph[Any, Any, Any, Any] | None = None,
    worker_fn: BacktestWorkerFn | None = None,
    risk_analyst_fn: RiskAnalystFn | None = None,
    paper_monitor_fn: PaperMonitorFn | None = None,
    build_context_fn: BuildContextFn | None = None,
    spawn_container_fn: SpawnContainerFn | None = None,
    schedule_wake_fn: ScheduleWakeFn | None = None,
    stop_container_fn: StopContainerFn | None = None,
    # ── Live subgraph seams (Stage 8g). Separate from the paper seams above
    # (e.g. live_registry_writer_fn ≠ paper's inline registry write) so the
    # live path's DB / credential / LLM / container effects are independently
    # stubbable. All default to their real implementations inside
    # build_live_subgraph when None.
    spawn_live_container_fn: SpawnLiveContainerFn | None = None,
    stop_live_container_fn: LiveStopContainerFn | None = None,
    stop_trading_fn: StopTradingFn | None = None,
    start_trading_fn: StartTradingFn | None = None,
    live_started_bump_fn: LiveStartedBumpFn | None = None,
    live_secrets_provider: SecretProvider | None = None,
    live_registry_writer_fn: RegistryWriterFn | None = None,
    build_snapshot_fn: BuildSnapshotFn | None = None,
    rationale_fn: RationaleFn | None = None,
    review_fn: PerformanceReviewFn | None = None,
    merge_fn: MergeFn | None = None,
    arbitrate_fn: ArbitrateFn | None = None,
    gate_audit_writer_fn: GateAuditWriterFn | None = None,
) -> CompiledStateGraph[StrategyState, StrategyState, StrategyState, StrategyState]:
    """Compile the per-strategy parent graph (BRD §5.2).

    Parameters
    ----------
    saver, store
        LangGraph saver + Store from the FastAPI lifespan (BRD §6.5).
    research_subgraph / validation_subgraph / paper_subgraph / live_subgraph
        Optional pre-built subgraphs. When None, the real subgraph is
        built from the leaf-effect seams below. Integration tests pass
        stub subgraphs to exercise parent routing + interrupt surfacing
        without Docker / real LLMs / cached data.
    worker_fn, risk_analyst_fn
        Validation subgraph seams (default: real backtest worker + real
        Opus risk_analyst).
    paper_monitor_fn, build_context_fn, spawn_container_fn,
    schedule_wake_fn, stop_container_fn
        Paper subgraph seams. ``schedule_wake_fn`` is supplied by the
        lifespan (the APScheduler-backed fn from Stage 7f); the others
        default to their real implementations.
    spawn_live_container_fn, stop_live_container_fn, stop_trading_fn,
    start_trading_fn, live_started_bump_fn, live_secrets_provider,
    live_registry_writer_fn, build_snapshot_fn, rationale_fn, review_fn,
    merge_fn, arbitrate_fn, gate_audit_writer_fn
        Live subgraph seams (Stage 8g, BRD §5.6). Doubled from the paper
        set where the effect differs (e.g. ``live_registry_writer_fn`` vs
        paper's inline registry write; live secrets vs paper secrets).
        ``start_trading_fn`` + ``live_started_bump_fn`` drive the D-5
        Option A live_pause-approve resume path. All default to their real
        implementations inside build_live_subgraph.
    """
    research = research_subgraph or build_research_subgraph(store=store)
    validation = validation_subgraph or build_validation_subgraph(
        worker_fn or _default_backtest_worker_fn,
        risk_analyst_fn=risk_analyst_fn,
        checkpointer=None,  # nested under the parent's saver
    )
    paper = paper_subgraph or build_paper_subgraph(
        spawn_container_fn=spawn_container_fn,
        paper_monitor_fn=paper_monitor_fn,
        build_context_fn=build_context_fn,
        schedule_wake_fn=schedule_wake_fn,
        stop_container_fn=stop_container_fn,
        checkpointer=None,  # nested under the parent's saver
    )
    live = live_subgraph or build_live_subgraph(
        spawn_live_container_fn=spawn_live_container_fn,
        stop_live_container_fn=stop_live_container_fn,
        secrets_provider=live_secrets_provider,
        registry_writer_fn=live_registry_writer_fn,
        build_snapshot_fn=build_snapshot_fn,
        rationale_fn=rationale_fn,
        review_fn=review_fn,
        merge_fn=merge_fn,
        arbitrate_fn=arbitrate_fn,
        gate_audit_writer_fn=gate_audit_writer_fn,
        stop_trading_fn=stop_trading_fn,
        start_trading_fn=start_trading_fn,
        live_started_bump_fn=live_started_bump_fn,
        checkpointer=None,  # nested under the parent's saver
    )

    builder: StateGraph[StrategyState, StrategyState, StrategyState, StrategyState] = (
        StateGraph(StrategyState)
    )
    builder.add_node("research_subgraph", research)
    builder.add_node("validation_subgraph", validation)
    builder.add_node("paper_subgraph", paper)
    builder.add_node("live_subgraph", live)

    builder.add_edge(START, "research_subgraph")
    builder.add_conditional_edges(
        "research_subgraph",
        _route_after_research,
        {"validation": "validation_subgraph", "end": END},
    )
    builder.add_conditional_edges(
        "validation_subgraph",
        _route_after_validation,
        {"paper": "paper_subgraph", "end": END},
    )
    builder.add_conditional_edges(
        "paper_subgraph",
        _route_after_paper,
        {"live": "live_subgraph", "end": END},
    )
    builder.add_edge("live_subgraph", END)

    return builder.compile(checkpointer=saver, store=store)
