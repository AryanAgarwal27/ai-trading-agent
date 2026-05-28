"""Paper-vs-backtest distribution comparison (BRD §5.5, §10).

The paper_monitor agent's ``compare_to_backtest`` tool calls into here
to decide whether live paper-trade per-trade returns have drifted from
the backtest distribution that justified advancing the strategy.

Two signals (BRD §10 thresholds):

* **Two-sample Kolmogorov-Smirnov test** on per-trade return
  distributions. ``p < MAX_PAPER_VS_BACKTEST_KS_PVALUE`` (0.05) → the
  distributions differ at the 5% level.
* **Relative Sharpe deviation.**
  ``|sharpe_paper - sharpe_backtest| / |sharpe_backtest|`` exceeding
  ``MAX_PAPER_VS_BACKTEST_SHARPE_DEVIATION`` (0.30) → performance drift.

scipy is NOT a project dependency, so the KS p-value uses the standard
asymptotic Kolmogorov-distribution approximation (the same closed form
``scipy.stats.ks_2samp`` used by default for large samples, with the
Numerical-Recipes small-sample correction term). Accurate for n ≳ 20
per sample; a 30-day paper run easily clears that. For very small
samples the approximation is mildly conservative (slightly overstates
divergence), which is the safe direction for a gate that protects real
capital.
"""

from __future__ import annotations

import math
from typing import TypedDict

import numpy as np

from orchestrator.gates.thresholds import (
    MAX_PAPER_VS_BACKTEST_KS_PVALUE,
    MAX_PAPER_VS_BACKTEST_SHARPE_DEVIATION,
)


class ComparisonResult(TypedDict):
    """Output of :func:`compare_paper_to_backtest`.

    ``diverged`` is the headline boolean the agent reads; the component
    booleans + raw numbers let the agent (and the dashboard) explain
    *why* in its rationale.
    """

    ks_statistic: float
    p_value: float
    n_paper: int
    n_backtest: int
    sharpe_paper: float
    sharpe_backtest: float
    sharpe_deviation: float
    ks_diverged: bool
    sharpe_diverged: bool
    diverged: bool


def _sharpe(returns: np.ndarray) -> float:
    """Per-trade Sharpe (mean / sample-std). 0.0 when undefined.

    Not annualized — this is a per-trade ratio used only for the
    paper-vs-backtest *relative* deviation, so the annualization factor
    cancels and would only add noise.
    """
    if returns.size < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    if std == 0.0:
        return 0.0
    return float(returns.mean() / std)


def _kolmogorov_q(lam: float) -> float:
    """Q_ks(λ) = 2 Σ_{j≥1} (-1)^(j-1) e^(-2 j² λ²).

    The complementary Kolmogorov CDF (Numerical Recipes §14.3). Series
    converges fast; we cap at 100 terms and early-exit once terms are
    negligible. Returns a value in [0, 1] that is the asymptotic
    two-sided p-value.
    """
    if lam <= 0.0:
        return 1.0
    total = 0.0
    for j in range(1, 101):
        term = 2.0 * ((-1) ** (j - 1)) * math.exp(-2.0 * j * j * lam * lam)
        total += term
        if abs(term) < 1e-10:
            break
    return total


def ks_two_sample(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample KS test. Returns ``(D_statistic, p_value)``.

    Pure numpy. Empty input on either side returns ``(0.0, 1.0)`` — no
    evidence of divergence — which the caller treats as "not diverged".
    """
    a = np.sort(a)
    b = np.sort(b)
    n1 = a.size
    n2 = b.size
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    all_vals = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_vals, side="right") / n1
    cdf_b = np.searchsorted(b, all_vals, side="right") / n2
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    en = math.sqrt(n1 * n2 / (n1 + n2))
    # Numerical-Recipes effective-lambda with small-sample correction.
    p = _kolmogorov_q((en + 0.12 + 0.11 / en) * d)
    return d, max(0.0, min(1.0, p))


def compare_paper_to_backtest(
    paper_returns: list[float], backtest_returns: list[float]
) -> ComparisonResult:
    """Compare paper per-trade returns to backtest per-trade returns.

    Returns a :class:`ComparisonResult`. ``diverged`` is True if EITHER
    the KS test flags a distribution difference OR the relative Sharpe
    deviation exceeds the BRD §10 threshold — either is sufficient cause
    for the monitor to distrust the live behaviour.
    """
    paper = np.asarray(paper_returns, dtype=float)
    bt = np.asarray(backtest_returns, dtype=float)

    d, p = ks_two_sample(paper, bt)
    sharpe_paper = _sharpe(paper)
    sharpe_bt = _sharpe(bt)

    if paper.size < 2 or bt.size < 2:
        # Too few samples on one side to compute a meaningful per-trade
        # Sharpe. Divergence requires evidence; with no/one paper trade
        # there is none, so report zero deviation rather than treating a
        # forced-zero Sharpe as a maximal deviation against the baseline.
        sharpe_dev = 0.0
    elif sharpe_bt == 0.0:
        # No backtest baseline to deviate from. Treat equal-zero as no
        # deviation; any nonzero paper Sharpe against a zero baseline is
        # maximal relative deviation (capped at 1.0 for a clean number).
        sharpe_dev = 0.0 if sharpe_paper == 0.0 else 1.0
    else:
        sharpe_dev = abs(sharpe_paper - sharpe_bt) / abs(sharpe_bt)

    ks_diverged = p < MAX_PAPER_VS_BACKTEST_KS_PVALUE
    sharpe_diverged = sharpe_dev > MAX_PAPER_VS_BACKTEST_SHARPE_DEVIATION

    return ComparisonResult(
        ks_statistic=d,
        p_value=p,
        n_paper=int(paper.size),
        n_backtest=int(bt.size),
        sharpe_paper=sharpe_paper,
        sharpe_backtest=sharpe_bt,
        sharpe_deviation=sharpe_dev,
        ks_diverged=ks_diverged,
        sharpe_diverged=sharpe_diverged,
        diverged=ks_diverged or sharpe_diverged,
    )
