"""Unit tests for the paper-vs-backtest comparison (Stage 7d).

Pure-function tests — no markers, run in the default CI invocation.
Verify the KS test + Sharpe-deviation logic against known shapes.
"""

from __future__ import annotations

import numpy as np

from orchestrator.tools.compare import (
    compare_paper_to_backtest,
    ks_two_sample,
)


def test_identical_distributions_do_not_diverge() -> None:
    """Same sample on both sides → high p-value, not diverged."""
    rng = np.random.default_rng(42)
    sample = rng.normal(0.01, 0.05, size=300).tolist()

    result = compare_paper_to_backtest(sample, sample)

    assert result["p_value"] > 0.05
    assert not result["ks_diverged"]
    assert not result["diverged"]
    # Identical samples → zero Sharpe deviation.
    assert result["sharpe_deviation"] == 0.0


def test_shifted_distribution_diverges_on_ks() -> None:
    """A clear location shift trips the KS test (p < 0.05)."""
    rng = np.random.default_rng(7)
    backtest = rng.normal(0.02, 0.04, size=300).tolist()
    # Paper returns shifted strongly negative — different distribution.
    paper = rng.normal(-0.03, 0.04, size=300).tolist()

    result = compare_paper_to_backtest(paper, backtest)

    assert result["p_value"] < 0.05
    assert result["ks_diverged"]
    assert result["diverged"]


def test_sharpe_deviation_flag_independent_of_ks() -> None:
    """Same-shape distributions with a big Sharpe gap flag sharpe_diverged.

    Scale the paper returns so the distribution overlaps enough that KS
    may not fire, but the per-trade Sharpe differs by more than 30%.
    """
    rng = np.random.default_rng(11)
    # Backtest: healthy positive Sharpe.
    backtest = rng.normal(0.02, 0.02, size=400).tolist()
    # Paper: near-zero mean, same spread → Sharpe collapses.
    paper = rng.normal(0.0, 0.02, size=400).tolist()

    result = compare_paper_to_backtest(paper, backtest)

    # The Sharpe deviation should be large (paper Sharpe ~0 vs backtest ~1).
    assert result["sharpe_deviation"] > 0.30
    assert result["sharpe_diverged"]
    assert result["diverged"]


def test_empty_paper_returns_is_not_diverged() -> None:
    """No paper trades yet → no evidence of divergence."""
    backtest = [0.01, 0.02, -0.01, 0.03]

    result = compare_paper_to_backtest([], backtest)

    assert result["n_paper"] == 0
    assert result["p_value"] == 1.0
    assert not result["diverged"]


def test_ks_two_sample_empty_returns_neutral() -> None:
    """ks_two_sample on an empty side returns (0.0, 1.0)."""
    d, p = ks_two_sample(np.asarray([]), np.asarray([1.0, 2.0, 3.0]))
    assert d == 0.0
    assert p == 1.0


def test_ks_pvalue_in_unit_interval() -> None:
    """p-value is always clamped to [0, 1] for adversarial inputs."""
    rng = np.random.default_rng(99)
    a = rng.normal(0, 1, size=50)
    b = rng.normal(5, 1, size=50)  # huge separation → tiny p
    _d, p = ks_two_sample(a, b)
    assert 0.0 <= p <= 1.0
