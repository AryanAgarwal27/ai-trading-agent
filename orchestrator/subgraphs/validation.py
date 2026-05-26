"""Validation subgraph — parallel backtests + robustness (BRD §5.4).

Built up stage-by-stage:

- **Stage 4b** — Send fan-out skeleton. ``plan_backtests`` emits one Send
  per (param_set × fold), workers run in parallel under the reducer
  contract, the worker function is injected via factory.

- **Stage 4c** (this commit) — anchored 6-fold walk-forward planner
  (BRD §5.4); real ``run_backtest`` wired as the worker via a factory
  closure; ``aggregate_results`` collapses per-fold ``BacktestResult``
  rows into per-param-set summary stats; ``gate_backtest`` applies the
  BRD §10 hard thresholds and writes the verdict to
  ``state["gate_decisions"]["backtest"]``. Failure routing
  (``Command(goto="archive")``) lands in Stage 4d once the archive node
  exists.

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

import statistics
from collections.abc import Awaitable, Callable
from datetime import date
from operator import add
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from orchestrator.gates import thresholds
from orchestrator.state import BacktestResult

# BaseCheckpointSaver is generic on the serializer type; we don't constrain
# it here, so accept any concrete saver (InMemorySaver for tests,
# AsyncPostgresSaver for production).
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

    # ─── Shared inputs (passed to every Send) ──────────────────────
    strategy_path: str
    pairs: list[str]
    timeframe: str

    # ─── Plan inputs — supplied by the caller before invoke ────────
    param_sets: list[dict[str, Any]]
    folds: list[dict[str, Any]]

    # ─── Per-Send transients ──────────────────────────────────────
    # Set inside the Send body, not via reducer. The underscore prefix is
    # the convention BRD §6.3 uses to mark them.
    _param_set: dict[str, Any]
    _fold: dict[str, Any]

    # ─── Reducer-aggregated outputs ───────────────────────────────
    # Send fan-out is incorrect without this — parallel writes overwrite.
    backtest_results: Annotated[list[BacktestResult], add]

    # ─── Gate verdicts ────────────────────────────────────────────
    # Single dict overwritten by aggregate_results + gate_backtest. Stage
    # 4d will add the "robustness" sub-dict; Stage 4e will add "paper".
    gate_decisions: dict[str, Any]


# ════════════════════════════════════════════════════════════════════════
# Walk-forward planning (BRD §5.4)
# ════════════════════════════════════════════════════════════════════════


def plan_walk_forward(
    *,
    data_start: date,
    train_months: int = 4,
    test_months: int = 1,
    n_folds: int = 6,
    anchored: bool = True,
) -> list[dict[str, Any]]:
    """Return the ``folds`` list for an anchored walk-forward (BRD §5.4).

    "Anchored 6-fold walk-forward (4 months train / 1 month test, sliding
    by 1 month)" decomposes into:

      - fold 1: train = [data_start, data_start + 4mo); test = [+4mo, +5mo)
      - fold 2: train = [data_start, data_start + 5mo); test = [+5mo, +6mo)
      - ...
      - fold 6: train = [data_start, data_start + 9mo); test = [+9mo, +10mo)

    The "anchored" property is that ``train`` always starts at
    ``data_start``; only the train end and the test window slide. Setting
    ``anchored=False`` produces a rolling walk-forward (fixed-size train
    sliding alongside test) for future comparison — BRD currently requires
    anchored, so the default is True.

    Each returned fold dict has::

        {
            "fold_id": "fold_<i>",
            "timerange": "<YYYYMMDD-YYYYMMDD>",  # OOS window
            "train_timerange": "<YYYYMMDD-YYYYMMDD>",  # IS window
        }

    Stage 4c's backtest worker uses ``timerange`` (the OOS window) as the
    Freqtrade ``--timerange`` argument. Stage 4d's monte_carlo + regime
    workers may consume ``train_timerange`` separately. The fold dict is
    kept JSON-shaped (no Python date objects) so it serializes cleanly
    through the Send payload and the Postgres checkpointer.
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be ≥ 1; got {n_folds}")
    if train_months < 1 or test_months < 1:
        raise ValueError(
            f"train_months and test_months must each be ≥ 1; got "
            f"train={train_months}, test={test_months}"
        )

    folds: list[dict[str, Any]] = []
    for i in range(n_folds):
        train_start = data_start if anchored else _add_months(data_start, i)
        train_end = _add_months(data_start, train_months + i)
        test_start = train_end
        test_end = _add_months(test_start, test_months)
        folds.append(
            {
                "fold_id": f"fold_{i + 1}",
                "timerange": f"{test_start.strftime('%Y%m%d')}-{test_end.strftime('%Y%m%d')}",
                "train_timerange": (
                    f"{train_start.strftime('%Y%m%d')}-{train_end.strftime('%Y%m%d')}"
                ),
            }
        )
    return folds


def _add_months(d: date, months: int) -> date:
    """Add ``months`` to ``d`` with end-of-month clamping.

    Avoids the ``dateutil`` dep for one helper. We don't need calendar
    precision — walk-forward folds are coarse-grained and Freqtrade's
    ``--timerange`` uses whole days.
    """
    total = d.month - 1 + months
    new_year = d.year + total // 12
    new_month = total % 12 + 1
    # Clamp day-of-month so 31 Jan + 1 month = 28/29 Feb.
    new_day = min(d.day, _days_in_month(new_year, new_month))
    return date(new_year, new_month, new_day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - date(year, month, 1)).days


# ════════════════════════════════════════════════════════════════════════
# Send fan-out router (BRD §6.3)
# ════════════════════════════════════════════════════════════════════════


def plan_backtests(state: ValidationState) -> list[Send]:
    """Router: emit one ``Send`` per (param_set × fold) combination.

    BRD §6.3 specifies the literal pattern — the Send carries the full
    state plus per-iteration extras (``_param_set``, ``_fold``); the
    worker function reads them and returns ``{"backtest_results":
    [result]}`` so the reducer concatenates outputs across parallel
    workers.

    Returns an empty list when either dimension is empty (the conditional
    edge handles this gracefully by routing straight to the next node).
    """
    param_sets = state.get("param_sets") or []
    folds = state.get("folds") or []
    return [
        Send("backtest_worker", {**state, "_param_set": ps, "_fold": fold})
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


# ════════════════════════════════════════════════════════════════════════
# Aggregation + gate (BRD §5.4 aggregate_results + gate_backtest)
# ════════════════════════════════════════════════════════════════════════


def aggregate_results(state: ValidationState) -> dict[str, Any]:
    """Collapse per-fold ``BacktestResult`` rows into per-param-set stats.

    For each ``param_set_id``, computes:

      - ``sharpe_is`` — mean of per-fold ``is_sharpe`` (Stage 4c proxy for
        IS performance; Stage 5+ split adds true IS/OOS separation).
      - ``min_sharpe_per_fold`` — worst fold's IS Sharpe. BRD §10
        ``MIN_OOS_SHARPE_PER_FOLD`` will gate against this in Stage 4d.
      - ``profit_factor`` — mean across folds.
      - ``max_dd`` — worst (largest) per-fold drawdown.
      - ``trades`` — total across folds.
      - ``oos_sharpe_mean`` — currently 0.0 because Stage 4c's single-fold
        runner always sets ``oos_sharpe=0.0``. Stage 5+ fills this in.
      - ``oos_ratio`` — ``oos_sharpe_mean / sharpe_is`` if both non-zero;
        else 0.0. Stage 4c can't really test this until OOS is wired.

    Writes ``state["gate_decisions"]["backtest"]["param_sets"]`` as a list
    of these summary dicts, plus the "best" pick by mean IS Sharpe.
    """
    results = state.get("backtest_results") or []
    by_ps: dict[str, list[BacktestResult]] = {}
    for r in results:
        by_ps.setdefault(r["param_set_id"], []).append(r)

    summaries: list[dict[str, Any]] = []
    for ps_id, rs in by_ps.items():
        is_sharpes = [r["is_sharpe"] for r in rs]
        profit_factors = [r["profit_factor"] for r in rs]
        max_dds = [r["max_dd"] for r in rs]
        trades = [r["trades"] for r in rs]
        oos_sharpes = [r["oos_sharpe"] for r in rs]

        sharpe_is_mean = statistics.fmean(is_sharpes) if is_sharpes else 0.0
        oos_sharpe_mean = statistics.fmean(oos_sharpes) if oos_sharpes else 0.0

        summaries.append(
            {
                "param_set_id": ps_id,
                "fold_count": len(rs),
                "sharpe_is": sharpe_is_mean,
                "min_sharpe_per_fold": min(is_sharpes) if is_sharpes else 0.0,
                "profit_factor": statistics.fmean(profit_factors) if profit_factors else 0.0,
                "max_dd": max(max_dds) if max_dds else 0.0,
                "trades": sum(trades),
                "oos_sharpe_mean": oos_sharpe_mean,
                "oos_ratio": (
                    oos_sharpe_mean / sharpe_is_mean
                    if sharpe_is_mean > 0 and oos_sharpe_mean != 0.0
                    else 0.0
                ),
            }
        )

    # Best = highest mean IS Sharpe. Tie-breaker is param_set_id alphabetical
    # for determinism — important because the integration test asserts on
    # the chosen set.
    best = max(summaries, key=lambda s: (s["sharpe_is"], s["param_set_id"]), default=None)

    existing_gates = state.get("gate_decisions") or {}
    return {
        "gate_decisions": {
            **existing_gates,
            "backtest": {
                "param_sets": summaries,
                "best_param_set_id": best["param_set_id"] if best else None,
            },
        }
    }


def gate_backtest(state: ValidationState) -> dict[str, Any]:
    """Apply BRD §10 hard thresholds to the aggregated best param set.

    Failure routing (``Command(goto="archive")``) lands in Stage 4d once
    the archive node exists. For 4c we only compute the verdict and
    record the failing thresholds in
    ``state["gate_decisions"]["backtest"]["failures"]``; the parent graph
    can read this and decide to advance or archive.

    A strategy passes the gate when ALL of the following hold for the
    best param set:

      - trades ≥ ``MIN_TRADES_IS``
      - sharpe_is ≥ ``MIN_SHARPE_IS``
      - profit_factor ≥ ``MIN_PROFIT_FACTOR_IS``
      - max_dd ≤ ``MAX_DRAWDOWN_IS``

    OOS thresholds (``MIN_OOS_TRADES``, ``MIN_OOS_RATIO``,
    ``MIN_OOS_SHARPE_PER_FOLD``, ``MIN_OOS_PROFIT_FACTOR``,
    ``MAX_OOS_DRAWDOWN``) are not checked here because Stage 4c's
    single-fold worker always reports ``oos_sharpe=0.0``. Stage 5+ wires
    OOS and these checks light up.
    """
    gates = state.get("gate_decisions") or {}
    backtest_block = gates.get("backtest") or {}
    summaries = backtest_block.get("param_sets") or []
    if not summaries:
        return {
            "gate_decisions": {
                **gates,
                "backtest": {
                    **backtest_block,
                    "passed": False,
                    "failures": ["no_param_set_summaries"],
                },
            }
        }

    best_id = backtest_block.get("best_param_set_id")
    best = next((s for s in summaries if s["param_set_id"] == best_id), summaries[0])

    failures: list[str] = []
    if best["trades"] < thresholds.MIN_TRADES_IS:
        failures.append(f"trades={best['trades']} < MIN_TRADES_IS={thresholds.MIN_TRADES_IS}")
    if best["sharpe_is"] < thresholds.MIN_SHARPE_IS:
        failures.append(
            f"sharpe_is={best['sharpe_is']:.3f} < " f"MIN_SHARPE_IS={thresholds.MIN_SHARPE_IS}"
        )
    if best["profit_factor"] < thresholds.MIN_PROFIT_FACTOR_IS:
        failures.append(
            f"profit_factor={best['profit_factor']:.3f} < "
            f"MIN_PROFIT_FACTOR_IS={thresholds.MIN_PROFIT_FACTOR_IS}"
        )
    if best["max_dd"] > thresholds.MAX_DRAWDOWN_IS:
        failures.append(
            f"max_dd={best['max_dd']:.3f} > MAX_DRAWDOWN_IS={thresholds.MAX_DRAWDOWN_IS}"
        )

    return {
        "gate_decisions": {
            **gates,
            "backtest": {
                **backtest_block,
                "passed": not failures,
                "failures": failures,
            },
        }
    }


# ════════════════════════════════════════════════════════════════════════
# Subgraph builder
# ════════════════════════════════════════════════════════════════════════


def build_validation_subgraph(
    worker_fn: BacktestWorkerFn,
    *,
    checkpointer: CheckpointSaver | None = None,
) -> CompiledStateGraph[ValidationState, ValidationState, ValidationState, ValidationState]:
    """Compile the Stage 4c validation subgraph.

    Pipeline: ``plan_backtests`` (router) → [N parallel ``backtest_worker``
    Sends] → ``aggregate_results`` → ``gate_backtest`` → END.

    Stage 4d will add the post-gate routing (passing strategies go to
    ``plan_robustness``; failing to ``archive``).

    Parameters
    ----------
    worker_fn
        Async function that maps a Send payload to a ``BacktestResult``.
        Stage 4c integration tests inject a closure around ``run_backtest``;
        Stage 4b unit tests inject a deterministic fake.
    checkpointer
        Optional saver. Unit tests pass ``InMemorySaver``; production
        wires the ``AsyncPostgresSaver`` from the FastAPI lifespan.
    """

    async def backtest_worker(payload: dict[str, Any]) -> dict[str, Any]:
        result = await worker_fn(payload)
        return {"backtest_results": [result]}

    builder: StateGraph[ValidationState, ValidationState, ValidationState, ValidationState] = (
        StateGraph(ValidationState)
    )
    builder.add_node("plan_backtests", _planner_passthrough)
    # async closure node — LangGraph 1.x typing of add_node infers Never for
    # the input generic when the node is a closure-captured async function,
    # so the strict-mypy arg-type check is a false positive here.
    builder.add_node("backtest_worker", backtest_worker)  # type: ignore[arg-type]
    builder.add_node("aggregate_results", aggregate_results)
    builder.add_node("gate_backtest", gate_backtest)

    builder.add_edge(START, "plan_backtests")
    builder.add_conditional_edges("plan_backtests", plan_backtests, ["backtest_worker"])
    builder.add_edge("backtest_worker", "aggregate_results")
    builder.add_edge("aggregate_results", "gate_backtest")
    builder.add_edge("gate_backtest", END)

    return builder.compile(checkpointer=checkpointer)
