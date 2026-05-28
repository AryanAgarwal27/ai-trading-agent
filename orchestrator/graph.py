"""Per-strategy parent graph (BRD §5.2).

Composes the lifecycle subgraphs in sequence. Stage 7g wires the full
research → validation → paper topology; the live subgraph (Stage 8)
slots in after paper graduates.

Topology (post-7g)::

    START
      │
      ▼
    research_subgraph ──archived──> END
      │ pass
      ▼
    validation_subgraph ──archived──> END
      │ stage="paper" (paper_gate approved)
      ▼
    paper_subgraph ──> END   (exits at stage="live" or stage="archived")

Each subgraph is a nested compiled graph added as a parent node. A
dynamic ``interrupt()`` inside a nested subgraph (paper_gate in
validation; paper_wait + live_gate in paper) DOES surface to the
parent's ``aget_state(config).tasks[*].interrupts`` — verified
empirically in Stage 7g — so the FastAPI ``GET /threads`` endpoint sees
pending interrupts through the real parent graph. The parent task NAME
for a nested interrupt is the SUBGRAPH NODE NAME (e.g.
``validation_subgraph``), NOT the inner gate name, so the resume
endpoints (``/approve``, ``/wake``) identify the gate by the interrupt
payload's ``"kind"`` field rather than the task name (see
orchestrator/main.py).

Inter-subgraph routing keys off ``state["stage"]``:
- research sets ``stage="archived"`` on its failure paths; a pass leaves
  stage unchanged → route to validation.
- validation's paper_gate approve sets ``stage="paper"`` → route to
  paper; any archived path → END.
- paper exits at ``stage="live"`` (live_gate approved) or
  ``stage="archived"`` (kill / reject); both terminal → END.

Every subgraph is injectable (``research_subgraph`` /
``validation_subgraph`` / ``paper_subgraph``) so integration tests can
compose with lightweight stub subgraphs and exercise the parent routing
+ interrupt surfacing without Docker, real LLMs, or cached market data.
Production passes none of these and the real subgraphs are built from
the leaf-effect seams (``worker_fn``, ``risk_analyst_fn``,
``paper_monitor_fn``, ``spawn_container_fn``, ``schedule_wake_fn`` …).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.postgres.aio import AsyncPostgresStore

from orchestrator.state import BacktestResult, StrategyState
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


def build_per_strategy_graph(
    saver: AsyncPostgresSaver | Any,
    store: AsyncPostgresStore | Any,
    *,
    research_subgraph: CompiledStateGraph[Any, Any, Any, Any] | None = None,
    validation_subgraph: CompiledStateGraph[Any, Any, Any, Any] | None = None,
    paper_subgraph: CompiledStateGraph[Any, Any, Any, Any] | None = None,
    worker_fn: BacktestWorkerFn | None = None,
    risk_analyst_fn: RiskAnalystFn | None = None,
    paper_monitor_fn: PaperMonitorFn | None = None,
    build_context_fn: BuildContextFn | None = None,
    spawn_container_fn: SpawnContainerFn | None = None,
    schedule_wake_fn: ScheduleWakeFn | None = None,
    stop_container_fn: StopContainerFn | None = None,
) -> CompiledStateGraph[StrategyState, StrategyState, StrategyState, StrategyState]:
    """Compile the per-strategy parent graph (BRD §5.2).

    Parameters
    ----------
    saver, store
        LangGraph saver + Store from the FastAPI lifespan (BRD §6.5).
    research_subgraph / validation_subgraph / paper_subgraph
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

    builder: StateGraph[StrategyState, StrategyState, StrategyState, StrategyState] = (
        StateGraph(StrategyState)
    )
    builder.add_node("research_subgraph", research)
    builder.add_node("validation_subgraph", validation)
    builder.add_node("paper_subgraph", paper)

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
    builder.add_edge("paper_subgraph", END)

    return builder.compile(checkpointer=saver, store=store)
