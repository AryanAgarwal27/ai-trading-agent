"""Gate thresholds — single source of truth (BRD §10).

**Do not put thresholds anywhere else.** Every gate check in the validation,
paper, and live subgraphs reads from this module. If you find yourself
copying a literal value into a node, stop and import it from here instead —
the BRD calls this out by name as a non-negotiable rule.

Values below are verbatim from BRD §10. SPEC.md §2 confirms no v1 overrides;
re-tuning happens here (and only here) per BRD §10's "do not put thresholds
anywhere else" rule, after the first 10 strategies complete a lifecycle.

Layout mirrors BRD §10's section ordering so a grep for a threshold name
finds the BRD prose and this module side-by-side.
"""

from __future__ import annotations

# ─── Backtest hard gate (in-sample, anchored 6-fold walk-forward) ───────
# Failing any of these routes the strategy to archive before robustness runs.

MIN_TRADES_IS: int = 150
"""Minimum total IS trades across all folds. Fewer = statistically uninformative."""

MIN_OOS_TRADES: int = 30
"""Minimum total OOS trades across all folds. Below this, OOS Sharpe is noise."""

MIN_SHARPE_IS: float = 1.5
"""IS Sharpe floor. Below this the strategy isn't worth the OOS check."""

MIN_PROFIT_FACTOR_IS: float = 1.5
"""IS profit factor floor (gross wins / gross losses)."""

MAX_DRAWDOWN_IS: float = 0.20
"""Max IS drawdown as positive fraction. 0.20 = 20%."""

# ─── OOS / walk-forward gate ────────────────────────────────────────────

MIN_OOS_RATIO: float = 0.6
"""``mean(OOS Sharpe) / mean(IS Sharpe)`` floor. Below 0.6 = overfit signal."""

MIN_OOS_SHARPE_PER_FOLD: float = 0.0
"""No fold may lose money on its OOS slice. Per-fold check, not average."""

MIN_OOS_PROFIT_FACTOR: float = 1.2
"""OOS profit factor floor."""

MAX_OOS_DRAWDOWN: float = 0.25
"""Max OOS drawdown as positive fraction. 0.25 = 25%."""

# ─── Robustness gate ─────────────────────────────────────────────────────
# Cheap deterministic gate that runs BEFORE the LLM risk_analyst. Per
# operator decision (Stage 4 handoff): failing strategies route to archive
# without burning Opus tokens.

MIN_MC_5TH_PERCENTILE_RETURN: float = 0.0
"""Trade-level bootstrap (1000 iters): 5th-percentile final equity must be ≥ 0."""

MIN_REGIMES_PASSED: int = 2
"""Of 3 vol regimes (low/mid/high), strategy must clear the bar in ≥ 2."""

MAX_FEE_STRESS_DEGRADATION_2X: float = 0.40
"""Doubling exchange fee may degrade Sharpe by at most 40% relative."""

MAX_FEE_STRESS_DEGRADATION_3X: float = 0.60
"""Tripling exchange fee may degrade Sharpe by at most 60% relative."""

# ─── Paper gate (advisory — human decides at HITL) ───────────────────────
# These are NOT auto-fail thresholds. The dashboard surfaces them as
# context for the human at `paper_gate`; the operator approves/rejects.

MIN_PAPER_DAYS: int = 30
"""BRD §1.1 rule 4: paper ≥ 30 days before live, non-negotiable."""

MAX_PAPER_VS_BACKTEST_KS_PVALUE: float = 0.05
"""KS-test on per-trade returns; below this advise reject (distributions diverge)."""

MAX_PAPER_VS_BACKTEST_SHARPE_DEVIATION: float = 0.30
"""Paper Sharpe within ±30% of backtest Sharpe — advisory."""

# ─── Live monitoring — AUTO PAUSE, not advisory ──────────────────────────
# These fire the kill switch (BRD §11) without human intervention. The
# graph re-syncs on next wake and surfaces `live_pause_review` to HITL.

KILL_SWITCH_DRAWDOWN: float = 0.12
"""Global drawdown from running peak. ``POST /api/v1/stop`` at 12%."""

KILL_SWITCH_CONSECUTIVE_LOSSES: int = 10
"""Hard stop after 10 consecutive losing trades."""

DAILY_LOSS_LIMIT_PCT: float = 0.03
"""-3% rolling 24h → ``POST /api/v1/stopbuy`` (graceful, lets opens run)."""

MAX_OPEN_TRADES: int = 4
"""Concurrent position cap; written into Freqtrade config."""

MAX_POSITION_CONCENTRATION: float = 0.30
"""One pair ≤ 30% of equity; enforced via Freqtrade protection."""

# ─── Live capital ────────────────────────────────────────────────────────

LIVE_CAPITAL_CAP_USD: int = 500
"""SPEC §1 Q3: $500 live cap. Re-tuning gated on SPEC §4.2 criteria."""
