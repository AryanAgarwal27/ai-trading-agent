"""Validation subgraph — parallel backtests + robustness (BRD §5.4).

Built up stage-by-stage:

- **Stage 4b** (this commit) — Send fan-out skeleton. ``plan_backtests``
  emits one Send per (param_set × fold) per BRD §6.3, workers run in
  parallel under the reducer contract, and the subgraph returns once all
  Sends complete. The worker function is **injected via factory** so the
  unit test wires a deterministic fake and the integration test (4c) wires
  the real ``run_backtest``.

- **Stage 4c** — replace the fake worker wiring with ``run_backtest``;
  add anchored 6-fold walk-forward planner; add ``aggregate_results`` +
  ``gate_backtest`` after the fan-out.

- **Stage 4d** — robustness fan-out (``plan_robustness`` →
  ``monte_carlo_worker`` + ``regime_worker`` + ``fee_stress_worker`` +
  ``gate_robustness``). Per operator decision: ``gate_robustness`` is a
  cheap deterministic check that runs BEFORE the LLM risk_analyst so
  failing strategies route to archive without burning Opus tokens.

- **Stage 4e** — ``risk_analyst`` ReAct node (Opus 4.7) + ``paper_gate``
  interrupt stub. Real HITL gate lands in Stage 6.

The pluggable worker contract (``BacktestWorkerFn``) is the key
testability seam: 4b's unit test injects a deterministic fake to verify
the Send/reducer wiring without ever launching Docker; 4c's integration
test injects a closure around ``run_backtest`` to exercise the real
Freqtrade subprocess path.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from operator import add
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from orchestrator.state import BacktestResult

# BaseCheckpointSaver is generic on the serializer type; we don't constrain
# it here, so accept any concrete saver (InMemorySaver for tests, AsyncPostgresSaver
# for production).
CheckpointSaver = BaseCheckpointSaver[Any]

# ─── Pluggable worker contract ──────────────────────────────────────────
# Async because the real worker (Stage 4c) wraps asyncio.to_thread on the
# Freqtrade subprocess and we want the orchestrator's event loop to schedule
# other Sends while one is in-flight.
BacktestWorkerFn = Callable[[dict[str, Any]], Awaitable[BacktestResult]]


class ValidationState(TypedDict, total=False):
    """Workspace state for the validation subgraph.

    ``total=False`` so the parent graph can hand the subgraph a partial
    dict and the subgraph populates fields as it runs. The reducer field
    is the BRD §6.3 contract — parallel Send workers MUST write via
    ``Annotated[..., operator.add]`` or the second write overwrites the
    first and the test catches it.
    """

    # Plan inputs — supplied by the caller before invoke.
    param_sets: list[dict[str, Any]]
    folds: list[dict[str, Any]]

    # Per-Send transients — set inside the Send body, not via reducer. The
    # underscore prefix is the convention BRD §6.3 uses to mark them.
    _param_set: dict[str, Any]
    _fold: dict[str, Any]

    # Reducer-aggregated outputs. Send fan-out is incorrect without this.
    backtest_results: Annotated[list[BacktestResult], add]


def plan_backtests(state: ValidationState) -> list[Send]:
    """Router: emit one ``Send`` per (param_set × fold) combination.

    BRD §6.3 specifies the literal pattern — the Send carries the
    per-iteration extras (``_param_set``, ``_fold``); the worker function
    reads them and returns ``{"backtest_results": [result]}`` so the
    reducer concatenates outputs across parallel workers.

    Returns an empty list when either dimension is empty (the conditional
    edge handles this gracefully by routing straight to the next node).
    """
    param_sets = state.get("param_sets") or []
    folds = state.get("folds") or []
    return [
        Send("backtest_worker", {"_param_set": ps, "_fold": fold})
        for ps in param_sets
        for fold in folds
    ]


def _planner_passthrough(state: ValidationState) -> dict[str, Any]:
    """Source node for the conditional fan-out edge.

    The actual fan-out happens via ``plan_backtests`` attached as a
    conditional-edge function; this node exists so the conditional edge
    has a stable source vertex to attach from. It does not mutate state.
    """
    return {}


def build_validation_subgraph(
    worker_fn: BacktestWorkerFn,
    *,
    checkpointer: CheckpointSaver | None = None,
) -> CompiledStateGraph[ValidationState, ValidationState, ValidationState, ValidationState]:
    """Compile the Stage 4b fan-out skeleton.

    Parameters
    ----------
    worker_fn
        Async function that maps a Send payload (``{"_param_set":...,
        "_fold":...}``) to a ``BacktestResult``. Injected so 4b unit tests
        use a deterministic fake; 4c integration tests wire the real
        ``run_backtest``.
    checkpointer
        Optional saver. Unit tests pass an ``InMemorySaver``; production
        wires the ``AsyncPostgresSaver`` from the FastAPI lifespan
        (BRD §6.5).
    """

    async def backtest_worker(payload: dict[str, Any]) -> dict[str, Any]:
        result = await worker_fn(payload)
        # Single-element list goes through the Annotated[list, add] reducer
        # on ValidationState.backtest_results — concatenates across all
        # parallel Sends. Returning a non-list here would break the contract.
        return {"backtest_results": [result]}

    builder: StateGraph[ValidationState, ValidationState, ValidationState, ValidationState] = (
        StateGraph(ValidationState)
    )
    builder.add_node("plan_backtests", _planner_passthrough)
    # async closure node — LangGraph 1.x typing of add_node infers Never for
    # the input generic when the node is a closure-captured async function,
    # so the strict-mypy arg-type check is a false positive here.
    builder.add_node("backtest_worker", backtest_worker)  # type: ignore[arg-type]
    builder.add_edge(START, "plan_backtests")
    builder.add_conditional_edges("plan_backtests", plan_backtests, ["backtest_worker"])
    builder.add_edge("backtest_worker", END)

    return builder.compile(checkpointer=checkpointer)
