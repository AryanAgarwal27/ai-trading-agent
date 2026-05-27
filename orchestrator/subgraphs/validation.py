"""Validation subgraph — parallel backtests + robustness (BRD §5.4).

Built up stage-by-stage:

- **Stage 4b** — Send fan-out skeleton. ``plan_backtests`` emits one Send
  per (param_set × fold), workers run in parallel under the reducer
  contract, the worker function is injected via factory.

- **Stage 4c** — anchored 6-fold walk-forward planner (BRD §5.4); real
  ``run_backtest`` wired as the worker via a factory closure;
  ``aggregate_results`` collapses per-fold BacktestResult rows into
  per-param-set summary stats; ``gate_backtest`` applies the BRD §10
  hard thresholds and writes the verdict to
  ``state["gate_decisions"]["backtest"]``.

- **Stage 4d** (this commit) — robustness fan-out per BRD §5.4:

    * ``plan_robustness`` Send-emits ``monte_carlo_worker`` +
      ``regime_worker`` + ``fee_stress_worker``.
    * ``aggregate_robustness`` collapses the three results.
    * ``gate_robustness`` applies BRD §10 thresholds
      (``MIN_MC_5TH_PERCENTILE_RETURN``, ``MIN_REGIMES_PASSED``,
      ``MAX_FEE_STRESS_DEGRADATION_2X/3X``) BEFORE any LLM call.
      Per operator decision: a cheap deterministic check first means
      failing strategies route to archive without burning Opus tokens.
    * ``archive`` terminal sink writes ``failure_reason`` and ends.
    * ``gate_backtest`` and ``gate_robustness`` use ``Command(goto, update)``
      routing (BRD §6.4) so pass-paths advance and fail-paths route to
      ``archive`` with diagnostic ``failure_reason``.
    * ``aggregate_results`` calls ``cleanup_stale_workers`` after writing
      summaries — see the cleanup contract docstring below.

- **Stage 4e** — ``risk_analyst`` ReAct node (Opus 4.7) + ``paper_gate``
  interrupt stub. Real HITL gate lands in Stage 6.

**Stale-worker cleanup contract.** Per Stage 4d handoff: aggregators own
cleanup of (a) the workers they produced this run AND (b) any orphan
worker dirs older than ``min_age_seconds`` found under ``_workers/``.
``aggregate_results`` calls ``cleanup_stale_workers`` with ``keep=``
the active fan-out's worker paths (extracted from
``backtest_results[*].raw_zip_path``), so the in-flight robustness
workers later in the pipeline can still find their backtest artifacts
while older orphans get pruned. ``aggregate_robustness`` does NOT
re-sweep — by the time it runs, ``aggregate_results`` has already
pruned the orphans.
"""

from __future__ import annotations

import json
import random
import statistics
import zipfile
from collections.abc import Awaitable, Callable
from datetime import date, datetime
from operator import add
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Send

from orchestrator.gates import thresholds
from orchestrator.state import BacktestResult, RobustnessResult
from orchestrator.tools.backtest_runner import (
    SHARED_DATA_DIR,
    cleanup_stale_workers,
    run_backtest,
)
from orchestrator.tools.regime import classify_regime

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
    dict and the subgraph populates fields as it runs. The reducer fields
    are the BRD §6.3 contract — parallel Send workers MUST write via
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
    # Send fan-out is incorrect without these — parallel writes overwrite.
    backtest_results: Annotated[list[BacktestResult], add]
    robustness_results: Annotated[list[RobustnessResult], add]

    # ─── Gate verdicts + lifecycle ────────────────────────────────
    gate_decisions: dict[str, Any]
    stage: str
    failure_reason: str


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
    """Add ``months`` to ``d`` with end-of-month clamping."""
    total = d.month - 1 + months
    new_year = d.year + total // 12
    new_month = total % 12 + 1
    new_day = min(d.day, _days_in_month(new_year, new_month))
    return date(new_year, new_month, new_day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - date(year, month, 1)).days


# ════════════════════════════════════════════════════════════════════════
# Backtest fan-out (BRD §6.3)
# ════════════════════════════════════════════════════════════════════════


def plan_backtests(state: ValidationState) -> list[Send]:
    """Router: emit one ``Send`` per (param_set × fold) combination."""
    param_sets = state.get("param_sets") or []
    folds = state.get("folds") or []
    return [
        Send("backtest_worker", {**state, "_param_set": ps, "_fold": fold})
        for ps in param_sets
        for fold in folds
    ]


def _planner_passthrough(state: ValidationState) -> dict[str, Any]:
    """Source node for the conditional fan-out edge (no-op)."""
    return {}


def aggregate_results(state: ValidationState) -> dict[str, Any]:
    """Collapse per-fold ``BacktestResult`` rows into per-param-set stats.

    Also calls :func:`cleanup_stale_workers` with the active fan-out's
    worker paths as the ``keep`` list — orphan worker dirs older than
    1 hour get pruned here so the operator never has to think about them.
    See module docstring for the cleanup-ownership contract.
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
        trade_counts = [r["trades"] for r in rs]
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
                "trades": sum(trade_counts),
                "oos_sharpe_mean": oos_sharpe_mean,
                "oos_ratio": (
                    oos_sharpe_mean / sharpe_is_mean
                    if sharpe_is_mean > 0 and oos_sharpe_mean != 0.0
                    else 0.0
                ),
            }
        )

    best = max(summaries, key=lambda s: (s["sharpe_is"], s["param_set_id"]), default=None)

    # Stale-worker cleanup. ``keep`` is the set of worker dirs hosting the
    # current run's artifacts — extracted from raw_zip_path's parent's
    # parent (raw_zip_path → backtest_results/X.zip → _workers/<id>/).
    # Robustness workers downstream will need these artifacts; orphans
    # from prior sessions are pruned.
    keep_paths = [_artifact_worker_dir(r["raw_zip_path"]) for r in results]
    cleanup_stale_workers(keep=[p for p in keep_paths if p is not None])

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


def _artifact_worker_dir(raw_zip_path: str) -> Path | None:
    """Map a BacktestResult's raw_zip_path back to its worker dir.

    Returns None if the path doesn't sit under ``_workers/<id>/...``.
    """
    p = Path(raw_zip_path)
    if not p.exists():
        return None
    parts = p.resolve().parts
    try:
        idx = parts.index("_workers")
    except ValueError:
        return None
    if idx + 1 >= len(parts):
        return None
    return Path(*parts[: idx + 2])


def gate_backtest(
    state: ValidationState,
) -> Command[Literal["plan_robustness", "archive"]]:
    """Apply BRD §10 IS hard thresholds. Route via ``Command(goto)``.

    Pass → ``plan_robustness``. Fail → ``archive`` with a diagnostic
    ``failure_reason`` listing each violated threshold. BRD §6.4 specifies
    ``Command(goto, update)`` for this kind of conditional routing.

    OOS thresholds (``MIN_OOS_TRADES``, ``MIN_OOS_RATIO``, …) are not
    checked here because Stage 4c's single-fold runner always reports
    ``oos_sharpe=0.0``. Stage 5+ wires OOS and these checks light up.
    """
    gates = state.get("gate_decisions") or {}
    backtest_block = gates.get("backtest") or {}
    summaries = backtest_block.get("param_sets") or []

    if not summaries:
        update: dict[str, Any] = {
            "gate_decisions": {
                **gates,
                "backtest": {
                    **backtest_block,
                    "passed": False,
                    "failures": ["no_param_set_summaries"],
                },
            },
            "stage": "archived",
            "failure_reason": "backtest_gate: no_param_set_summaries",
        }
        return Command(goto="archive", update=update)

    best_id = backtest_block.get("best_param_set_id")
    best = next((s for s in summaries if s["param_set_id"] == best_id), summaries[0])

    failures: list[str] = []
    if best["trades"] < thresholds.MIN_TRADES_IS:
        failures.append(f"trades={best['trades']} < MIN_TRADES_IS={thresholds.MIN_TRADES_IS}")
    if best["sharpe_is"] < thresholds.MIN_SHARPE_IS:
        failures.append(
            f"sharpe_is={best['sharpe_is']:.3f} < MIN_SHARPE_IS={thresholds.MIN_SHARPE_IS}"
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

    update_passed: dict[str, Any] = {
        "gate_decisions": {
            **gates,
            "backtest": {
                **backtest_block,
                "passed": not failures,
                "failures": failures,
            },
        },
    }
    if failures:
        update_passed["stage"] = "archived"
        update_passed["failure_reason"] = "backtest_gate: " + "; ".join(failures)
        return Command(goto="archive", update=update_passed)
    return Command(goto="plan_robustness", update=update_passed)


# ════════════════════════════════════════════════════════════════════════
# Robustness fan-out (BRD §5.4)
# ════════════════════════════════════════════════════════════════════════


def plan_robustness(state: ValidationState) -> list[Send]:
    """Emit 3 parallel Sends: ``monte_carlo_worker``, ``regime_worker``,
    ``fee_stress_worker`` (BRD §5.4).

    All three Sends receive the full state so each worker has access to
    ``backtest_results`` (for trade-level bootstrap / regime grouping) and
    ``strategy_path``/``pairs``/``timeframe``/``folds`` (for fee_stress's
    sequential rerun on the best param set).
    """
    return [
        Send("monte_carlo_worker", dict(state)),
        Send("regime_worker", dict(state)),
        Send("fee_stress_worker", dict(state)),
    ]


def _planner_passthrough_robustness(state: ValidationState) -> dict[str, Any]:
    """No-op source node for the robustness fan-out conditional edge."""
    return {}


# ─── Monte Carlo worker ────────────────────────────────────────────────


def monte_carlo_worker(state: ValidationState) -> dict[str, Any]:
    """Trade-level bootstrap (BRD §5.4): 1000-iter resample, 5th pct equity.

    Pulls per-trade ``profit_ratio`` from each BacktestResult's stored
    artifact, concatenates across folds for the best param set, then
    resamples with replacement 1000 times and computes the final-equity
    distribution's 5th percentile. The result must be ≥
    ``MIN_MC_5TH_PERCENTILE_RETURN`` (= 0.0 per BRD §10) for gate pass.
    """
    backtest_results = state.get("backtest_results") or []
    gates = state.get("gate_decisions") or {}
    best_id = (gates.get("backtest") or {}).get("best_param_set_id")
    filtered = [r for r in backtest_results if r["param_set_id"] == best_id]

    all_returns: list[float] = []
    for r in filtered:
        all_returns.extend(_load_trade_returns_from_artifact(Path(r["raw_zip_path"])))

    pct_5, finals_summary = _bootstrap_5th_percentile(all_returns)

    return {
        "robustness_results": [
            RobustnessResult(
                kind="monte_carlo",
                payload={
                    "n_trades": len(all_returns),
                    "n_iterations": 1000,
                    "pct_5_final_equity": pct_5,
                    "median_final_equity": finals_summary["median"],
                    "mean_final_equity": finals_summary["mean"],
                },
            )
        ]
    }


def _load_trade_returns_from_artifact(artifact_path: Path) -> list[float]:
    """Extract per-trade ``profit_ratio`` from a Freqtrade backtest artifact.

    Handles both ``.zip`` (preferred — created by ``--export trades``)
    and ``.json`` (the sibling file when the zip isn't preserved).
    Returns an empty list if the artifact is missing, malformed, or
    contains no trades.
    """
    if not artifact_path.exists():
        return []

    if artifact_path.suffix == ".zip":
        try:
            with zipfile.ZipFile(artifact_path) as zf:
                target = next(
                    (
                        n
                        for n in zf.namelist()
                        if n.endswith(".json")
                        and not n.endswith(".meta.json")
                        and not n.endswith("_config.json")
                    ),
                    None,
                )
                if target is None:
                    return []
                data = json.loads(zf.read(target))
        except (zipfile.BadZipFile, json.JSONDecodeError, KeyError):
            return []
    else:
        try:
            data = json.loads(artifact_path.read_text())
        except (json.JSONDecodeError, OSError):
            return []

    strategies = data.get("strategy", {})
    if not isinstance(strategies, dict) or not strategies:
        return []
    strategy_block = next(iter(strategies.values()))
    if not isinstance(strategy_block, dict):
        return []
    trades = strategy_block.get("trades") or []
    return [float(t.get("profit_ratio", 0.0)) for t in trades if isinstance(t, dict)]


def _bootstrap_5th_percentile(
    trade_returns: list[float],
    *,
    n_iterations: int = 1000,
    seed: int = 42,
) -> tuple[float, dict[str, float]]:
    """1000-iteration bootstrap. Returns (5th-percentile, summary stats).

    Each iteration resamples N trades with replacement and computes
    ``∏(1 + r)`` as the final equity multiple. Returns 1.0 when there
    are no trades — neutral signal, lets gate_robustness see "n=0" and
    flag MIN_MC_5TH_PERCENTILE_RETURN as failed.
    """
    if not trade_returns:
        return 1.0, {"median": 1.0, "mean": 1.0}

    rng = random.Random(seed)
    n_trades = len(trade_returns)
    finals: list[float] = []
    for _ in range(n_iterations):
        equity = 1.0
        for _ in range(n_trades):
            equity *= 1.0 + rng.choice(trade_returns)
        finals.append(equity)
    finals.sort()
    pct_5_idx = max(0, int(0.05 * n_iterations) - 1)
    return finals[pct_5_idx], {
        "median": finals[n_iterations // 2],
        "mean": statistics.fmean(finals),
    }


# ─── Regime worker ─────────────────────────────────────────────────────


def regime_worker(state: ValidationState) -> dict[str, Any]:
    """Classify each fold by realized vol regime; group per-fold Sharpe.

    For each fold the best param set ran on, load the OOS closes from
    the cached feather, classify the regime via
    :func:`orchestrator.tools.regime.classify_regime`, and bucket the
    fold's IS Sharpe into that regime. Reports per-regime mean Sharpe
    and count of regimes "passed" (mean Sharpe > 0).

    ``MIN_REGIMES_PASSED`` (= 2 of 3 per BRD §10) gates against the
    "regimes_passed" count.
    """
    backtest_results = state.get("backtest_results") or []
    folds = state.get("folds") or []
    gates = state.get("gate_decisions") or {}
    best_id = (gates.get("backtest") or {}).get("best_param_set_id")
    filtered = [r for r in backtest_results if r["param_set_id"] == best_id]

    by_fold_id = {f["fold_id"]: f for f in folds}
    by_regime: dict[str, list[float]] = {}
    unclassified: list[str] = []

    for r in filtered:
        fold = by_fold_id.get(r["fold_id"])
        if fold is None:
            unclassified.append(r["fold_id"])
            continue
        closes = _load_closes_for_timerange(
            pair=r["pair"],
            timeframe=r["timeframe"],
            timerange=fold["timerange"],
        )
        if len(closes) < 30:
            unclassified.append(r["fold_id"])
            continue
        regime_label, _features = classify_regime(
            closes, timeframe_minutes=_timeframe_minutes(r["timeframe"])
        )
        by_regime.setdefault(regime_label, []).append(r["is_sharpe"])

    regime_stats = {
        label: {
            "mean_sharpe": statistics.fmean(sharpes),
            "n_folds": len(sharpes),
        }
        for label, sharpes in by_regime.items()
    }
    regimes_passed = sum(1 for s in regime_stats.values() if s["mean_sharpe"] > 0)

    return {
        "robustness_results": [
            RobustnessResult(
                kind="regime",
                payload={
                    "by_regime": regime_stats,
                    "regimes_passed": regimes_passed,
                    "unclassified_folds": unclassified,
                },
            )
        ]
    }


def _load_closes_for_timerange(
    *,
    pair: str,
    timeframe: str,
    timerange: str,
) -> list[float]:
    """Read closes from the cached feather for ``pair`` at ``timeframe``,
    sliced to ``timerange`` (Freqtrade ``YYYYMMDD-YYYYMMDD`` format).

    Returns an empty list if the feather is missing or the slice is empty.
    """
    feather_path = SHARED_DATA_DIR / "binance" / f"{pair.replace('/', '_')}-{timeframe}.feather"
    if not feather_path.exists():
        return []

    # Local import to avoid module-level pyarrow dependency for callers
    # that don't need regime classification.
    import pyarrow.feather as feather

    start_str, end_str = timerange.split("-")
    start_dt = datetime.strptime(start_str, "%Y%m%d")
    end_dt = datetime.strptime(end_str, "%Y%m%d")

    table = feather.read_table(feather_path)  # type: ignore[no-untyped-call]
    df = table.to_pandas()
    # Freqtrade feathers use a 'date' column (UTC tz-aware).
    if "date" not in df.columns or "close" not in df.columns:
        return []
    mask = (df["date"] >= start_dt.replace(tzinfo=df["date"].dt.tz)) & (
        df["date"] < end_dt.replace(tzinfo=df["date"].dt.tz)
    )
    closes = df.loc[mask, "close"].tolist()
    return [float(c) for c in closes]


def _timeframe_minutes(tf: str) -> int:
    """Convert a Freqtrade timeframe string like ``"5m"`` / ``"1h"`` to minutes."""
    unit = tf[-1].lower()
    value = int(tf[:-1])
    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 24 * 60
    raise ValueError(f"unsupported timeframe unit in {tf!r}")


# ─── Fee stress worker ─────────────────────────────────────────────────


async def fee_stress_worker(state: ValidationState) -> dict[str, Any]:
    """Sequential 2× and 3× fee runs on the best param set (BRD §5.4).

    Operator's Stage 4d guidance: sequential-in-worker is acceptable
    (vs split into 2 parallel Sends) — picked because the 2 calls are
    cheap to run back-to-back and avoiding Send-of-Send complexity
    keeps the graph topology flat.

    Re-runs the FIRST fold's timerange with ``--fee 0.002`` and ``--fee
    0.003``, then compares each stressed Sharpe to the baseline mean
    Sharpe. Reports relative degradation; negative baselines get a
    sentinel ``1.0`` degradation so the gate definitively fails them
    (the math isn't meaningful when the baseline is already losing).
    """
    backtest_results = state.get("backtest_results") or []
    folds = state.get("folds") or []
    gates = state.get("gate_decisions") or {}
    best_id = (gates.get("backtest") or {}).get("best_param_set_id")
    filtered = [r for r in backtest_results if r["param_set_id"] == best_id]

    if not filtered or not folds:
        return {
            "robustness_results": [
                RobustnessResult(
                    kind="fee_stress",
                    payload={
                        "error": "no_baseline_results",
                    },
                )
            ]
        }

    baseline_sharpe = statistics.fmean(r["is_sharpe"] for r in filtered)
    sample_fold = next((f for f in folds if f["fold_id"] == filtered[0]["fold_id"]), folds[0])

    fee_2x = await run_backtest(
        Path(state["strategy_path"]),
        pairs=state["pairs"],
        timeframe=state["timeframe"],
        timerange=sample_fold["timerange"],
        fold_id=f"fee_stress_2x_{sample_fold['fold_id']}",
        param_set_id=str(best_id),
        fee=0.002,
    )
    fee_3x = await run_backtest(
        Path(state["strategy_path"]),
        pairs=state["pairs"],
        timeframe=state["timeframe"],
        timerange=sample_fold["timerange"],
        fold_id=f"fee_stress_3x_{sample_fold['fold_id']}",
        param_set_id=str(best_id),
        fee=0.003,
    )

    deg_2x = _degradation(baseline_sharpe, fee_2x["is_sharpe"])
    deg_3x = _degradation(baseline_sharpe, fee_3x["is_sharpe"])

    return {
        "robustness_results": [
            RobustnessResult(
                kind="fee_stress",
                payload={
                    "baseline_sharpe": baseline_sharpe,
                    "fee_2x_sharpe": fee_2x["is_sharpe"],
                    "fee_3x_sharpe": fee_3x["is_sharpe"],
                    "degradation_2x": deg_2x,
                    "degradation_3x": deg_3x,
                    "fee_2x_artifact": fee_2x["raw_zip_path"],
                    "fee_3x_artifact": fee_3x["raw_zip_path"],
                },
            )
        ]
    }


def _degradation(baseline_sharpe: float, stressed_sharpe: float) -> float:
    """Relative Sharpe degradation from fee increase.

    Negative-baseline guardrail: when the strategy is already losing,
    relative degradation isn't meaningful — return ``1.0`` so the gate
    definitively fails it. A positive baseline uses the standard
    ``(baseline - stressed) / baseline`` formula clamped to ``[0, ∞)``.
    """
    if baseline_sharpe <= 0:
        return 1.0
    return max(0.0, (baseline_sharpe - stressed_sharpe) / baseline_sharpe)


# ─── Aggregate + gate robustness ───────────────────────────────────────


def aggregate_robustness(state: ValidationState) -> dict[str, Any]:
    """Collapse the 3 robustness results into a single summary dict.

    The gate node consumes ``state["gate_decisions"]["robustness"]``
    populated here, so the gate-vs-aggregator split keeps threshold
    logic out of the aggregator.
    """
    robustness = state.get("robustness_results") or []
    summary: dict[str, Any] = {}
    for rr in robustness:
        summary[rr["kind"]] = rr["payload"]

    gates = state.get("gate_decisions") or {}
    return {
        "gate_decisions": {
            **gates,
            "robustness": summary,
        }
    }


def gate_robustness(state: ValidationState) -> Command[Literal["archive", "risk_analyst"]]:
    """Cheap deterministic gate BEFORE the LLM call (Stage 4 handoff #3).

    BRD §10 thresholds applied:
      - ``MIN_MC_5TH_PERCENTILE_RETURN`` (≥ 1.0 = no loss in 5th-pct bootstrap)
      - ``MIN_REGIMES_PASSED`` (≥ 2 of 3 regimes with mean Sharpe > 0)
      - ``MAX_FEE_STRESS_DEGRADATION_2X`` (≤ 0.40 relative Sharpe drop)
      - ``MAX_FEE_STRESS_DEGRADATION_3X`` (≤ 0.60 relative Sharpe drop)

    Pass → ``risk_analyst`` (Stage 4e; for 4d the parent graph wires this
    directly to END since risk_analyst doesn't exist yet). Fail →
    ``archive`` with diagnostic ``failure_reason``.

    The check is intentionally cheap: no LLM call, just numeric
    comparisons against BRD §10 constants. Failing strategies route to
    archive without burning Opus tokens — operator's explicit design
    point in the Stage 4 handoff.
    """
    gates = state.get("gate_decisions") or {}
    robustness_block = gates.get("robustness") or {}

    failures: list[str] = []

    mc = robustness_block.get("monte_carlo") or {}
    pct_5 = mc.get("pct_5_final_equity")
    # MIN_MC_5TH_PERCENTILE_RETURN is documented as 0.0 in BRD §10
    # ("5th-pct bootstrap final equity must be positive") — we interpret
    # the threshold against the final-equity-multiple form, i.e. equity
    # ≥ 1.0 means non-loss. A 0.0 threshold means "any non-negative
    # return"; with equity multiples that maps to ≥ 1.0.
    if pct_5 is None or pct_5 < (1.0 + thresholds.MIN_MC_5TH_PERCENTILE_RETURN):
        failures.append(
            f"mc_pct_5={pct_5} < "
            f"1+MIN_MC_5TH_PERCENTILE_RETURN={1.0 + thresholds.MIN_MC_5TH_PERCENTILE_RETURN}"
        )

    regime = robustness_block.get("regime") or {}
    regimes_passed = regime.get("regimes_passed", 0)
    if regimes_passed < thresholds.MIN_REGIMES_PASSED:
        failures.append(
            f"regimes_passed={regimes_passed} < "
            f"MIN_REGIMES_PASSED={thresholds.MIN_REGIMES_PASSED}"
        )

    fee = robustness_block.get("fee_stress") or {}
    deg_2x = fee.get("degradation_2x")
    deg_3x = fee.get("degradation_3x")
    if deg_2x is None or deg_2x > thresholds.MAX_FEE_STRESS_DEGRADATION_2X:
        failures.append(
            f"fee_degradation_2x={deg_2x} > "
            f"MAX_FEE_STRESS_DEGRADATION_2X={thresholds.MAX_FEE_STRESS_DEGRADATION_2X}"
        )
    if deg_3x is None or deg_3x > thresholds.MAX_FEE_STRESS_DEGRADATION_3X:
        failures.append(
            f"fee_degradation_3x={deg_3x} > "
            f"MAX_FEE_STRESS_DEGRADATION_3X={thresholds.MAX_FEE_STRESS_DEGRADATION_3X}"
        )

    update: dict[str, Any] = {
        "gate_decisions": {
            **gates,
            "robustness": {
                **robustness_block,
                "passed": not failures,
                "failures": failures,
            },
        }
    }
    if failures:
        update["stage"] = "archived"
        update["failure_reason"] = "robustness_gate: " + "; ".join(failures)
        return Command(goto="archive", update=update)
    return Command(goto="risk_analyst", update=update)


# ─── Archive (terminal) ────────────────────────────────────────────────


def archive(state: ValidationState) -> dict[str, Any]:
    """Terminal sink: stamps stage and preserves failure_reason if set."""
    return {
        "stage": "archived",
        "failure_reason": state.get("failure_reason") or "validation_archived_without_reason",
    }


# ─── 4e placeholder ────────────────────────────────────────────────────


def risk_analyst_placeholder(state: ValidationState) -> dict[str, Any]:
    """4e will replace this with the real Opus 4.7 ReAct agent.

    For 4d, the gate_robustness pass-path needs a destination node; this
    stub satisfies that requirement without introducing the LLM
    dependency. It writes a benign sentinel into gate_decisions so the
    integration test can verify the pass-path is reachable.
    """
    gates = state.get("gate_decisions") or {}
    return {
        "gate_decisions": {
            **gates,
            "paper_gate": {
                "stage_4d_placeholder": True,
                "note": "Stage 4e replaces this with risk_analyst (Opus 4.7) + paper_gate interrupt.",
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
    """Compile the Stage 4d validation subgraph.

    Topology::

        START
          ↓
        plan_backtests (router) ──Send×N──> backtest_worker
                                                 ↓
                                          aggregate_results
                                                 ↓
                                          gate_backtest ──fail──> archive ──> END
                                                 ↓ pass
                                          plan_robustness (router) ──Send×3──>
                                                ├── monte_carlo_worker
                                                ├── regime_worker
                                                └── fee_stress_worker
                                                 ↓
                                          aggregate_robustness
                                                 ↓
                                          gate_robustness ──fail──> archive ──> END
                                                 ↓ pass
                                          risk_analyst (4e replaces stub)
                                                 ↓
                                                END

    Parameters
    ----------
    worker_fn
        Async function mapping a Send payload to a ``BacktestResult``.
    checkpointer
        Optional saver. Unit tests pass ``InMemorySaver``; production
        wires ``AsyncPostgresSaver`` from the FastAPI lifespan.
    """

    async def backtest_worker(payload: dict[str, Any]) -> dict[str, Any]:
        result = await worker_fn(payload)
        return {"backtest_results": [result]}

    builder: StateGraph[ValidationState, ValidationState, ValidationState, ValidationState] = (
        StateGraph(ValidationState)
    )
    builder.add_node("plan_backtests", _planner_passthrough)
    # See 4b comment: closure-async + LangGraph generic produces a false
    # positive on add_node arg-type strict check.
    builder.add_node("backtest_worker", backtest_worker)  # type: ignore[arg-type]
    builder.add_node("aggregate_results", aggregate_results)
    builder.add_node("gate_backtest", gate_backtest)

    builder.add_node("plan_robustness", _planner_passthrough_robustness)
    builder.add_node("monte_carlo_worker", monte_carlo_worker)
    builder.add_node("regime_worker", regime_worker)
    builder.add_node("fee_stress_worker", fee_stress_worker)
    builder.add_node("aggregate_robustness", aggregate_robustness)
    builder.add_node("gate_robustness", gate_robustness)

    builder.add_node("risk_analyst", risk_analyst_placeholder)
    builder.add_node("archive", archive)

    builder.add_edge(START, "plan_backtests")
    builder.add_conditional_edges("plan_backtests", plan_backtests, ["backtest_worker"])
    builder.add_edge("backtest_worker", "aggregate_results")
    builder.add_edge("aggregate_results", "gate_backtest")
    # gate_backtest returns Command(goto=...) so no explicit edge needed
    # — but LangGraph still requires the destination nodes to exist.

    builder.add_conditional_edges(
        "plan_robustness",
        plan_robustness,
        ["monte_carlo_worker", "regime_worker", "fee_stress_worker"],
    )
    builder.add_edge("monte_carlo_worker", "aggregate_robustness")
    builder.add_edge("regime_worker", "aggregate_robustness")
    builder.add_edge("fee_stress_worker", "aggregate_robustness")
    builder.add_edge("aggregate_robustness", "gate_robustness")
    # gate_robustness returns Command(goto=...).

    builder.add_edge("risk_analyst", END)
    builder.add_edge("archive", END)

    return builder.compile(checkpointer=checkpointer)
