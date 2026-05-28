"""Stage 4b unit tests — Send fan-out + reducer correctness (BRD §6.3).

Two layers, both offline (no Docker, no Postgres):

1. ``test_plan_backtests_*`` — direct calls to the router function.
   Asserts the BRD §6.3 Send pattern: one Send per (param_set × fold),
   target node ``backtest_worker``, per-Send body carries ``_param_set``
   and ``_fold`` extras.

2. ``test_reducer_concatenates_five_parallel_worker_writes`` — the BRD
   correctness gate. Compiles the subgraph with an ``InMemorySaver`` and
   a deterministic fake worker that returns a tagged ``BacktestResult``
   per payload. Invokes with **5 param sets × 1 fold = 5 Sends**, and
   asserts the final state has exactly 5 distinct entries.

The 5-Send count matches BRD §13 Stage 4 DoD verbatim ("5 parallel Send
workers produce 5 BacktestResults that the reducer concatenates"). The
walk-forward fold count (6, per BRD §5.4) is a separate concern wired
in Stage 4c — don't conflate.

If the reducer is missing or wrong, the parallel writes overwrite each
other under LangGraph 1.x semantics and the assertion sees < 5 entries.
That's the failure mode this test guards against.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Send

from orchestrator.state import BacktestResult
from orchestrator.subgraphs.validation import (
    ValidationState,
    build_validation_subgraph,
    plan_backtests,
    prepare_validation_inputs,
)


def _make_param_sets(n: int) -> list[dict[str, Any]]:
    return [{"id": f"ps_{i}", "value": i * 0.1} for i in range(n)]


def _single_fold() -> list[dict[str, Any]]:
    return [{"fold_id": "fold_0", "timerange": "20240101-20240108"}]


# ─── Router-only tests (no graph) ─────────────────────────────────────────


def test_plan_backtests_emits_one_send_per_param_set_times_fold() -> None:
    state: ValidationState = {
        "param_sets": _make_param_sets(5),
        "folds": _single_fold(),
    }
    sends = plan_backtests(state)
    assert len(sends) == 5, "Should produce 5 Sends for 5 param_sets × 1 fold"
    assert all(isinstance(s, Send) for s in sends)
    assert all(s.node == "backtest_worker" for s in sends), "All Sends must target backtest_worker"
    # Per-Send body carries the right extras.
    seen_ids = {s.arg["_param_set"]["id"] for s in sends}
    assert seen_ids == {f"ps_{i}" for i in range(5)}
    assert all(s.arg["_fold"]["fold_id"] == "fold_0" for s in sends)


def test_plan_backtests_cartesian_product_two_by_three() -> None:
    """Cartesian product check: 2 param_sets × 3 folds = 6 Sends."""
    state: ValidationState = {
        "param_sets": _make_param_sets(2),
        "folds": [{"fold_id": f"f{i}", "timerange": "20240101-20240108"} for i in range(3)],
    }
    sends = plan_backtests(state)
    assert len(sends) == 6
    keys = {(s.arg["_param_set"]["id"], s.arg["_fold"]["fold_id"]) for s in sends}
    assert keys == {(f"ps_{i}", f"f{j}") for i in range(2) for j in range(3)}


def test_plan_backtests_is_empty_with_no_param_sets() -> None:
    """Empty inputs must not raise — the conditional edge routes to next node."""
    assert plan_backtests({"param_sets": [], "folds": _single_fold()}) == []
    assert plan_backtests({"folds": _single_fold()}) == []  # missing key
    assert plan_backtests({"param_sets": _make_param_sets(3)}) == []  # no folds


# ─── End-to-end subgraph test (compiled with InMemorySaver) ────────────────


@pytest.mark.asyncio
async def test_reducer_concatenates_five_parallel_worker_writes() -> None:
    """BRD §6.3 correctness gate.

    Without ``Annotated[list, operator.add]`` on
    ``ValidationState.backtest_results``, parallel writes from the 5 Sends
    would overwrite each other under LangGraph 1.x state-merge semantics
    and only the last writer's result would survive. This test asserts
    all 5 made it through.
    """

    async def fake_worker(payload: dict[str, Any]) -> BacktestResult:
        # Tagged so each result is distinct — lets us assert no Send was
        # silently dropped, even if the count looks right.
        ps = payload["_param_set"]
        fold = payload["_fold"]
        return BacktestResult(
            param_set_id=ps["id"],
            pair="BTC/USDT",
            timeframe="5m",
            fold_id=fold["fold_id"],
            is_sharpe=ps["value"],
            oos_sharpe=0.0,
            profit_factor=1.0,
            max_dd=0.0,
            trades=0,
            raw_zip_path=f"fake://{ps['id']}",
        )

    graph = build_validation_subgraph(fake_worker, checkpointer=InMemorySaver())

    initial: ValidationState = {
        "param_sets": _make_param_sets(5),
        "folds": _single_fold(),
    }
    thread_id = f"test_{uuid.uuid4().hex[:8]}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    final = await graph.ainvoke(initial, config=config)

    assert (
        "backtest_results" in final
    ), "Subgraph produced no backtest_results — reducer wired incorrectly?"
    results = final["backtest_results"]
    assert len(results) == 5, (
        f"Expected 5 results from 5 parallel Sends, got {len(results)}. "
        "If 1, the reducer is missing (BRD §6.3): "
        "ValidationState.backtest_results must be Annotated[list, operator.add]."
    )
    # Each tagged result came through; no Send was dropped or overwritten.
    seen = {r["param_set_id"] for r in results}
    assert seen == {f"ps_{i}" for i in range(5)}


def test_plan_backtests_empty_inputs_yields_no_sends() -> None:
    """Empty-fan-out safety at the router level (BRD §6.3).

    plan_backtests with empty param_sets (or empty folds) returns [] —
    LangGraph routes the conditional edge regardless of Send-list
    contents, so an empty list cleanly launches no worker. (As of Stage
    7h the validation SUBGRAPH can't reach plan_backtests with empty
    inputs — prepare_validation_inputs derives them — so this empty-fan-
    out property is asserted at the router level where it stays
    reachable.)
    """
    assert plan_backtests({"param_sets": [], "folds": _single_fold()}) == []
    assert plan_backtests({"param_sets": _make_param_sets(3), "folds": []}) == []
    assert plan_backtests({}) == []


def test_prepare_validation_inputs_derives_param_sets_and_folds() -> None:
    """Stage 7h adapter: research's single `params` → param_sets + folds.

    Mirrors what the parent graph hands in from a research run: a single
    `params` dict + strategy_path, no param_sets/folds. The adapter wraps
    params into a one-element param_sets (id from strategy_id) and
    computes the anchored 6-fold walk-forward.
    """
    updates = prepare_validation_inputs(
        {
            "strategy_id": "strat-123",
            "params": {"rsi_buy_threshold": 30, "ema_fast": 12},
            "strategy_path": "/tmp/strat.py",
        }
    )
    # One derived param set, id = strategy_id, params spread in.
    assert updates["param_sets"] == [
        {"id": "strat-123", "rsi_buy_threshold": 30, "ema_fast": 12}
    ]
    # Anchored 6-fold walk-forward (BRD §5.4).
    assert len(updates["folds"]) == 6
    assert all({"fold_id", "timerange"} <= set(f) for f in updates["folds"])
    # strategy_path already set → not overridden.
    assert "strategy_path" not in updates


def test_prepare_validation_inputs_respects_seeded_values() -> None:
    """A caller that seeds param_sets/folds (standalone validation tests)
    is unaffected — the adapter only derives when absent."""
    seeded = {
        "param_sets": _make_param_sets(2),
        "folds": _single_fold(),
        "strategy_path": "/tmp/x.py",
    }
    updates = prepare_validation_inputs(seeded)
    assert "param_sets" not in updates
    assert "folds" not in updates
    assert "strategy_path" not in updates


def test_prepare_validation_inputs_falls_back_to_generated_strategy_path() -> None:
    """strategy_path derives from artifacts.generated_strategy_path when
    the channel isn't directly set (parent-graph handoff resilience)."""
    updates = prepare_validation_inputs(
        {
            "strategy_id": "s1",
            "params": {},
            "artifacts": {"generated_strategy_path": "/tmp/gen.py"},
        }
    )
    assert updates["strategy_path"] == "/tmp/gen.py"
