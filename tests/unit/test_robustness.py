"""Stage 4d unit tests — robustness workers + gate_robustness + cleanup.

Five test groups:

1. ``test_bootstrap_*`` — Monte Carlo bootstrap math against synthetic
   trade-return arrays. No Docker, no Freqtrade artifact dependency.

2. ``test_degradation_*`` — fee-stress degradation formula, with the
   negative-baseline guardrail explicitly exercised.

3. ``test_gate_robustness_*`` — verdict logic on synthetic robustness
   summaries. Pure stateless, covers each BRD §10 threshold.

4. ``test_cleanup_stale_workers_*`` — orphan sweep against a tmp dir
   structured like ``_workers/``. Verifies (a) old orphans get pruned,
   (b) active ``keep`` paths are preserved, (c) recent dirs below
   ``min_age_seconds`` are preserved.

5. ``test_monte_carlo_worker_e2e_with_synthetic_artifact`` — end-to-end
   on the worker function, with a temp Freqtrade-shaped JSON artifact
   written to disk. Exercises ``_load_trade_returns_from_artifact``.
"""

from __future__ import annotations

import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any, cast

import pytest

from orchestrator.subgraphs.validation import (
    ValidationState,
    _bootstrap_5th_percentile,
    _degradation,
    _load_trade_returns_from_artifact,
    gate_robustness,
    monte_carlo_worker,
)
from orchestrator.tools.backtest_runner import cleanup_stale_workers

# ─── 1. Monte Carlo bootstrap math ──────────────────────────────────────


def test_bootstrap_5th_percentile_on_all_positive_returns_is_above_one() -> None:
    """All-positive trades: final equity is always > 1.0 regardless of resample.

    5th percentile of all-positive geometric products must also be > 1.0.
    """
    returns = [0.01, 0.02, 0.005, 0.015, 0.03]  # all positive
    pct_5, summary = _bootstrap_5th_percentile(returns, n_iterations=200, seed=1)
    assert pct_5 > 1.0
    assert summary["median"] > 1.0


def test_bootstrap_5th_percentile_on_all_negative_returns_is_below_one() -> None:
    """All-negative trades: final equity always < 1.0; 5th percentile too."""
    returns = [-0.01, -0.02, -0.005, -0.015, -0.03]
    pct_5, summary = _bootstrap_5th_percentile(returns, n_iterations=200, seed=1)
    assert pct_5 < 1.0
    assert summary["median"] < 1.0


def test_bootstrap_5th_percentile_returns_neutral_on_empty_input() -> None:
    """No trades → 1.0 (neutral). Lets gate_robustness fail with clear
    mc_pct_5 < threshold signal rather than crashing on empty input."""
    pct_5, summary = _bootstrap_5th_percentile([], n_iterations=100)
    assert pct_5 == 1.0
    assert summary == {"median": 1.0, "mean": 1.0}


def test_bootstrap_seed_is_deterministic() -> None:
    """Same seed → same output. Reproducibility for debugging."""
    returns = [0.01, -0.02, 0.005, -0.01, 0.02]
    a, _ = _bootstrap_5th_percentile(returns, n_iterations=100, seed=42)
    b, _ = _bootstrap_5th_percentile(returns, n_iterations=100, seed=42)
    assert a == b


# ─── 2. Fee-stress degradation math ────────────────────────────────────


def test_degradation_positive_baseline_relative_drop() -> None:
    """Standard case: baseline 2.0 Sharpe, stressed 1.5 → 25% degradation."""
    assert _degradation(2.0, 1.5) == pytest.approx(0.25)
    # No degradation = same Sharpe.
    assert _degradation(2.0, 2.0) == pytest.approx(0.0)
    # Negative stressed below baseline: clamped to relative drop (large).
    assert _degradation(2.0, -1.0) == pytest.approx(1.5)


def test_degradation_negative_baseline_returns_full_drop() -> None:
    """Negative baseline → sentinel 1.0 (definitive gate failure).

    Relative degradation isn't meaningful when baseline is already losing.
    BRD §10 thresholds presume a profitable baseline; this guardrail keeps
    a negative-Sharpe strategy from accidentally passing fee-stress just
    because the math underflows.
    """
    assert _degradation(-0.5, -0.6) == 1.0
    assert _degradation(0.0, 1.0) == 1.0  # zero baseline = boundary


def test_degradation_clamped_at_zero_when_stressed_better_than_baseline() -> None:
    """Stressed > baseline (unusual but possible): degradation clamped to 0."""
    assert _degradation(1.0, 1.5) == 0.0


# ─── 3. gate_robustness verdicts ───────────────────────────────────────


def _robustness_summary(
    *,
    mc_pct_5: float = 1.05,
    regimes_passed: int = 2,
    deg_2x: float = 0.20,
    deg_3x: float = 0.40,
) -> dict[str, object]:
    """Build a synthetic robustness gate_decisions block."""
    return {
        "monte_carlo": {"pct_5_final_equity": mc_pct_5, "n_trades": 100},
        "regime": {"regimes_passed": regimes_passed, "by_regime": {}},
        "fee_stress": {
            "degradation_2x": deg_2x,
            "degradation_3x": deg_3x,
            "baseline_sharpe": 1.6,
            "fee_2x_sharpe": 1.3,
            "fee_3x_sharpe": 1.0,
        },
    }


def test_gate_robustness_passes_when_all_thresholds_met() -> None:
    state = {"gate_decisions": {"robustness": _robustness_summary()}}
    cmd = gate_robustness(cast(ValidationState, state))
    update = cast(dict[str, Any], cmd.update)
    assert cmd.goto == "risk_analyst"
    assert update["gate_decisions"]["robustness"]["passed"] is True
    assert update["gate_decisions"]["robustness"]["failures"] == []


def test_gate_robustness_archives_on_mc_5th_percentile_failure() -> None:
    state = {
        "gate_decisions": {"robustness": _robustness_summary(mc_pct_5=0.95)}  # below 1.0
    }
    cmd = gate_robustness(cast(ValidationState, state))
    update = cast(dict[str, Any], cmd.update)
    assert cmd.goto == "archive"
    assert update["stage"] == "archived"
    failures = update["gate_decisions"]["robustness"]["failures"]
    assert any("mc_pct_5" in f for f in failures), failures


def test_gate_robustness_archives_on_regimes_passed_failure() -> None:
    state = {
        "gate_decisions": {"robustness": _robustness_summary(regimes_passed=1)},
    }
    cmd = gate_robustness(cast(ValidationState, state))
    update = cast(dict[str, Any], cmd.update)
    assert cmd.goto == "archive"
    failures = update["gate_decisions"]["robustness"]["failures"]
    assert any("regimes_passed" in f for f in failures)


def test_gate_robustness_archives_on_fee_2x_degradation() -> None:
    state = {
        "gate_decisions": {"robustness": _robustness_summary(deg_2x=0.50)},  # > 0.40 cap
    }
    cmd = gate_robustness(cast(ValidationState, state))
    update = cast(dict[str, Any], cmd.update)
    assert cmd.goto == "archive"
    failures = update["gate_decisions"]["robustness"]["failures"]
    assert any("fee_degradation_2x" in f for f in failures)


def test_gate_robustness_archives_on_fee_3x_degradation() -> None:
    state = {
        "gate_decisions": {"robustness": _robustness_summary(deg_3x=0.70)},  # > 0.60 cap
    }
    cmd = gate_robustness(cast(ValidationState, state))
    update = cast(dict[str, Any], cmd.update)
    assert cmd.goto == "archive"
    failures = update["gate_decisions"]["robustness"]["failures"]
    assert any("fee_degradation_3x" in f for f in failures)


def test_gate_robustness_collects_multiple_failures() -> None:
    """All thresholds failed → all failures reported (operator can see the picture)."""
    state = {
        "gate_decisions": {
            "robustness": _robustness_summary(
                mc_pct_5=0.50, regimes_passed=0, deg_2x=0.80, deg_3x=0.95
            ),
        },
    }
    cmd = gate_robustness(cast(ValidationState, state))
    update = cast(dict[str, Any], cmd.update)
    assert cmd.goto == "archive"
    failures = update["gate_decisions"]["robustness"]["failures"]
    assert len(failures) == 4


# ─── 4. cleanup_stale_workers ──────────────────────────────────────────


def test_cleanup_removes_old_orphans_keeps_active_and_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three workers: one active (keep), one fresh (< min_age), one old.
    Expect only the old one removed.
    """
    workers = tmp_path / "_workers"
    workers.mkdir()
    monkeypatch.setattr("orchestrator.tools.backtest_runner.WORKERS_DIR", workers)

    active = workers / "active123"
    fresh = workers / "fresh456"
    old = workers / "old789"
    for w in (active, fresh, old):
        w.mkdir()
        (w / "marker.txt").write_text("hi")

    # Backdate the old worker by 2 hours so it's well above min_age_seconds=3600.
    two_hours_ago = time.time() - 2 * 3600
    os.utime(old, (two_hours_ago, two_hours_ago))

    removed = cleanup_stale_workers(keep=[active], min_age_seconds=3600)

    assert removed == ["old789"]
    assert active.exists(), "active worker (keep list) was wrongly removed"
    assert fresh.exists(), "fresh worker (under min_age) was wrongly removed"
    assert not old.exists(), "old orphan was not removed"


def test_cleanup_handles_missing_workers_dir_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No _workers/ dir at all → empty list, no exception."""
    monkeypatch.setattr(
        "orchestrator.tools.backtest_runner.WORKERS_DIR", tmp_path / "does_not_exist"
    )
    assert cleanup_stale_workers() == []


# ─── 5. monte_carlo_worker end-to-end with synthetic artifact ──────────


def test_monte_carlo_worker_reads_trades_from_zipped_artifact(tmp_path: Path) -> None:
    """Write a Freqtrade-shaped JSON inside a .zip; assert the worker
    extracts trades and runs the bootstrap."""

    # Build a JSON blob shaped like Freqtrade's output.
    artifact_dir = tmp_path / "backtest_results"
    artifact_dir.mkdir()
    zip_path = artifact_dir / "backtest-result-2024-01-01_00-00-00.zip"
    json_body = {
        "strategy": {
            "MeanReversionTemplate": {
                "trades": [
                    {"profit_ratio": 0.01},
                    {"profit_ratio": -0.005},
                    {"profit_ratio": 0.02},
                    {"profit_ratio": -0.01},
                    {"profit_ratio": 0.015},
                ]
            }
        },
        "strategy_comparison": [],
    }
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("backtest-result-2024-01-01_00-00-00.json", json.dumps(json_body))

    # Sanity: helper alone returns the 5 profit_ratios.
    returns = _load_trade_returns_from_artifact(zip_path)
    assert returns == [0.01, -0.005, 0.02, -0.01, 0.015]

    state = {
        "backtest_results": [
            {
                "param_set_id": "test_set",
                "pair": "BTC/USDT",
                "timeframe": "5m",
                "fold_id": "f1",
                "is_sharpe": 1.0,
                "oos_sharpe": 0.0,
                "profit_factor": 1.5,
                "max_dd": 0.05,
                "trades": 5,
                "raw_zip_path": str(zip_path),
            },
        ],
        "gate_decisions": {"backtest": {"best_param_set_id": "test_set"}},
    }
    result = monte_carlo_worker(cast(ValidationState, state))
    [rr] = result["robustness_results"]
    assert rr["kind"] == "monte_carlo"
    payload = rr["payload"]
    assert payload["n_trades"] == 5
    assert payload["n_iterations"] == 1000
    assert isinstance(payload["pct_5_final_equity"], float)
    # Mostly-positive returns → 5th-percentile final equity should be ≈ 1.0 or above.
    assert payload["median_final_equity"] > 0.95


def test_load_trades_returns_empty_on_missing_artifact() -> None:
    assert _load_trade_returns_from_artifact(Path("/does/not/exist.zip")) == []


def test_load_trades_returns_empty_on_malformed_zip(tmp_path: Path) -> None:
    """A zip without the expected JSON entry yields [] instead of raising."""
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("not_a_result.txt", "garbage")
    assert _load_trade_returns_from_artifact(bad) == []
