"""Stage 4c integration test — 5 parallel real Freqtrade workers.

BRD §13 Stage 4 DoD pinned by this test:

  - ``backtest_results`` has 5 entries after fan-out (the reducer
    correctness check, now exercised on real Freqtrade output instead of
    the 4b fake worker),
  - ``aggregate_results`` collapses per-fold results into per-param-set
    summary stats,
  - ``gate_backtest`` writes a verdict (``passed`` boolean +
    ``failures`` list) under ``state["gate_decisions"]["backtest"]``.

The fan-out uses **5 fixed 1-week timeranges of cached BTC/USDT 5m data**
against the mean-reversion template defaults. This is the same surface
area Stage 3's integration test covers per-fold; Stage 4c adds the
parallel-Send + aggregate + gate path on top.

The test also records the wall-clock time and per-worker disk usage,
captured via the ``capsys`` fixture, so the operator can see real
numbers from the first 5-worker fan-out before greenlighting Stage 4d
(which adds 3 more robustness Sends on top).

Test is opt-in via ``integration`` + ``freqtrade`` markers (skipped by
CI's ``pytest -m "not integration"``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from orchestrator.state import BacktestResult
from orchestrator.subgraphs.validation import (
    ValidationState,
    build_validation_subgraph,
)
from orchestrator.tools.backtest_runner import (
    SHARED_DATA_DIR,
    WORKERS_DIR,
    cleanup_worker,
    run_backtest,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STRATEGY_PATH = REPO_ROOT / "strategy_templates" / "mean_reversion_template.py"
CACHED_BTC_5M = SHARED_DATA_DIR / "binance" / "BTC_USDT-5m.feather"


def _skip_if_missing_prereqs() -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker not on PATH; skipping freqtrade fan-out test")
    if not CACHED_BTC_5M.exists():
        pytest.skip(f"cached BTC/USDT 5m feather missing at {CACHED_BTC_5M}")
    result = subprocess.run(
        ["docker", "image", "inspect", "freqtradeorg/freqtrade:stable_freqai"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("freqtradeorg/freqtrade:stable_freqai image not found locally")


def _five_one_week_folds() -> list[dict[str, Any]]:
    """Return 5 consecutive 1-week timeranges ending ~60 days before today.

    Deterministic against the 730-day cache for any "today" inside the
    download window. Five folds = the BRD §13 Stage 4 DoD's "5 parallel
    Send workers" figure; this is NOT the BRD §5.4 6-fold walk-forward
    (that gets exercised by the unit test ``test_walk_forward.py`` and
    will be wired into integration tests in a later stage when it matters).
    """
    today = datetime.now(UTC).date()
    end = today.fromordinal(today.toordinal() - 60)
    folds: list[dict[str, Any]] = []
    for i in range(5):
        slice_end = end.fromordinal(end.toordinal() - 7 * i)
        slice_start = slice_end.fromordinal(slice_end.toordinal() - 7)
        folds.append(
            {
                "fold_id": f"fold_{i + 1}",
                "timerange": f"{slice_start.strftime('%Y%m%d')}-{slice_end.strftime('%Y%m%d')}",
                "train_timerange": "n/a",
            }
        )
    return folds


def _dir_size_bytes(p: Path) -> int:
    """Recursive on-disk size in bytes (counts all regular files)."""
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except FileNotFoundError:
                # Worker may have been cleaned mid-walk; skip.
                pass
    return total


def _format_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


@pytest.mark.integration
@pytest.mark.freqtrade
@pytest.mark.asyncio
async def test_five_parallel_real_freqtrade_workers_aggregate_and_gate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run 5 real Freqtrade backtests in parallel; verify the full path.

    Asserts:
      1. ``backtest_results`` has 5 entries (reducer + real workers OK).
      2. ``aggregate_results`` produces a summary for the param set.
      3. ``gate_backtest`` writes ``passed`` + ``failures`` under
         ``state["gate_decisions"]["backtest"]``.

    Reports (printed via capsys):
      - Wall-clock time for the full ``ainvoke``.
      - Per-worker dir size at completion.
      - Total disk consumed across all 5 workers.
      - Aggregate stats (sharpe_is, trades, max_dd) for the param set.
    """
    _skip_if_missing_prereqs()

    # Snapshot WORKERS_DIR contents BEFORE the run so we can identify which
    # worker dirs this test produced (vs leftovers from past sessions).
    pre_existing = set(WORKERS_DIR.iterdir()) if WORKERS_DIR.exists() else set()

    # Closure around run_backtest so the subgraph's worker_fn signature
    # (payload -> BacktestResult) matches and we don't have to thread
    # strategy_path/pairs/timeframe through the Send payload manually.
    async def real_worker(payload: dict[str, Any]) -> BacktestResult:
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

    graph = build_validation_subgraph(real_worker, checkpointer=InMemorySaver())

    initial: ValidationState = {
        "strategy_path": str(STRATEGY_PATH),
        "pairs": ["BTC/USDT"],
        "timeframe": "5m",
        "param_sets": [{"id": "mean_reversion_defaults"}],
        "folds": _five_one_week_folds(),
    }
    thread_id = f"test_{uuid.uuid4().hex[:8]}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    t0 = time.perf_counter()
    final = await graph.ainvoke(initial, config=config)
    elapsed = time.perf_counter() - t0

    # ─── Assert reducer + worker correctness ──────────────────────
    results = final.get("backtest_results") or []
    assert len(results) == 5, f"Expected 5 BacktestResults, got {len(results)}"
    fold_ids = {r["fold_id"] for r in results}
    assert fold_ids == {f"fold_{i + 1}" for i in range(5)}
    # Every result has the right param_set_id.
    assert all(r["param_set_id"] == "mean_reversion_defaults" for r in results)
    # Every result's raw zip exists on disk (artifact retention contract).
    for r in results:
        assert Path(r["raw_zip_path"]).exists(), f"missing artifact: {r['raw_zip_path']}"

    # ─── Assert aggregate_results populated gate_decisions ────────
    gates = final.get("gate_decisions") or {}
    backtest_block = gates.get("backtest") or {}
    summaries = backtest_block.get("param_sets") or []
    assert len(summaries) == 1, "exactly one param set was tested"
    summary = summaries[0]
    assert summary["param_set_id"] == "mean_reversion_defaults"
    assert summary["fold_count"] == 5
    assert summary["trades"] >= 0  # could be 0 on a quiet week; non-zero is bonus
    assert backtest_block["best_param_set_id"] == "mean_reversion_defaults"

    # ─── Assert gate_backtest wrote a verdict ─────────────────────
    assert "passed" in backtest_block
    assert "failures" in backtest_block
    assert isinstance(backtest_block["passed"], bool)
    assert isinstance(backtest_block["failures"], list)

    # ─── Report timing + dir sizes for the operator ───────────────
    new_workers = sorted(
        (
            p
            for p in (WORKERS_DIR.iterdir() if WORKERS_DIR.exists() else [])
            if p not in pre_existing
        ),
        key=lambda p: p.stat().st_mtime,
    )
    sizes_by_id = {p.name: _dir_size_bytes(p) for p in new_workers}
    total_bytes = sum(sizes_by_id.values())

    lines = [
        "─── stage 4c parallel fan-out metrics ───",
        f"  workers spawned     : {len(new_workers)}",
        f"  wall-clock elapsed  : {elapsed:.2f}s",
        f"  worker dirs total   : {_format_bytes(total_bytes)}",
        *[f"    {wid}: {_format_bytes(n)}" for wid, n in sizes_by_id.items()],
        "  aggregate          :",
        f"    sharpe_is        : {summary['sharpe_is']:.3f}",
        f"    min_per_fold     : {summary['min_sharpe_per_fold']:.3f}",
        f"    profit_factor    : {summary['profit_factor']:.3f}",
        f"    max_dd           : {summary['max_dd']:.3f}",
        f"    total_trades     : {summary['trades']}",
        "  gate verdict       :",
        f"    passed           : {backtest_block['passed']}",
        *[f"    failure          : {f}" for f in backtest_block["failures"]],
    ]
    block = "\n".join(lines)

    # Print via capsys (visible with pytest -s) AND persist to a side file
    # so the operator can read the numbers regardless of pytest capture
    # mode. capsys.readouterr() consumes the buffer below, hence the file.
    metrics_file = REPO_ROOT / "tests" / "integration" / "_last_4c_metrics.txt"
    metrics_file.write_text(block + "\n", encoding="utf-8")
    print("\n" + block)

    captured = capsys.readouterr()
    assert "stage 4c parallel fan-out metrics" in captured.out

    # ─── Cleanup ──────────────────────────────────────────────────
    # All 5 workers succeeded; per Stage 4 handoff #2, the aggregator
    # owns cleanup post-persist. This integration test acts as the
    # stand-in aggregator here.
    for p in new_workers:
        cleanup_worker(p)
