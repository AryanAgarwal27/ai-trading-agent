"""Gate thresholds — single source of truth (BRD §10).

**Do not put thresholds anywhere else.** Every gate check in the validation,
paper, and live subgraphs reads from this module. If you find yourself
copying a literal value into a node, stop and import it from here instead —
the BRD calls this out by name as a non-negotiable rule.

Values are BRD §10 defaults EXCEPT the backtest hard gate, which carries the
2026-06-08 research-grounded re-tune recorded in SPEC §2 (threshold overrides)
and SPEC §6 (change-log). BRD §10 explicitly delegates this: "These values are
operator-tunable in SPEC.md ... Re-tune after the first 10 strategies have
completed a lifecycle." ~10 strategies ran, ALL failed (backtest Sharpes −8 to
−105), so the gate was re-tuned to a realistic, honest bar. Re-tuning still
happens here (and only here) per BRD §10's "do not put thresholds anywhere
else" rule.

The re-tuned symbols are ``MIN_SHARPE_IS`` (1.5 → 0.5), ``MIN_PROFIT_FACTOR_IS``
(1.5 → 1.2), and the NEW cross-fold ``MIN_POSITIVE_FOLDS`` (4 of 6). See the
SPEC §6 2026-06-08 entry for the full rationale + research citation (Bailey &
López de Prado Deflated Sharpe; honest walk-forward Sharpes cluster ~0.33–1.2;
buy-and-hold benchmark; 4/6-fold consistency over a lucky average).

Layout mirrors BRD §10's section ordering so a grep for a threshold name
finds the BRD prose and this module side-by-side.
"""

from __future__ import annotations

# ─── Backtest hard gate (in-sample, anchored 6-fold walk-forward) ───────
# Failing any of these routes the strategy to archive before robustness runs.

MIN_TRADES_IS: int = 90
"""Minimum total IS trades across all folds. **Re-tuned 150 → 90 on 2026-06-09
(SPEC §2/§6).**

Operator decision (BRD §10 makes these values SPEC-tunable after the first ~10
strategies; same mechanism as the 2026-06-08 re-tune above). On the SPEC §1 Q2
four-pair universe (BTC/ETH/SOL/BNB) across six one-month OOS folds, a *selective*
regime-filtered long-only strategy — the only shape that stays positive through a
bearish anchored window — fires only ~90–135 times; reaching 150 would force
looser entries into low-quality chop, exactly the over-trading-into-fees failure
the earlier strategies showed (Sharpe −8 to −105). The statistical-significance
role the 150 floor carried is now covered by the newer consistency gates:
``MIN_TRADES_PER_FOLD`` (every fold is exercised — no silent zero-trade fold) and
``MIN_POSITIVE_FOLDS`` (4/6 folds must be independently positive — the anti-luck
check the 2026-06-08 re-tune added). 90 trades = a 15/fold average. Raise it back
once the pair universe widens or the anchored window includes a trending regime
that supports more quality trades. (The two regime-filtered templates that
motivated this were triaged on a NON-authoritative proxy walk-forward; real
validation via ``POST /strategies/validate`` is the authority — proxy Sharpe/PF
figures are deliberately NOT recorded here as fact.)"""

MIN_TRADES_PER_FOLD: int = 5
"""Minimum trades in ANY single fold. Guards Flag 1 (Stage 7h): if the
default walk-forward window falls outside the cached OHLCV range, early
folds can produce 0 trades silently — masked by a fat aggregate
``trades`` count. A zero-trade fold means the strategy was never tested
on that slice; failing the gate is safer than promoting on incomplete
evidence."""

MIN_OOS_TRADES: int = 30
"""Minimum total OOS trades across all folds. Below this, OOS Sharpe is noise."""

MIN_SHARPE_IS: float = 0.5
"""IS Sharpe floor. **Re-tuned 1.5 → 0.5 on 2026-06-08 (SPEC §2/§6).**

The original 1.5 was the wrong headline goal: the attached research (long-only
spot, 4 majors, 5m/1h) found Sharpe ≥ 1.5 across 6 OOS folds *improbable* and a
likely selection-bias artifact under the Deflated Sharpe Ratio (Bailey & López
de Prado, JPM 2014) given only 6×7 = 42 days OOS. Honest walk-forward Sharpes
cluster ~0.33–1.2 (rigorous frameworks ~0.33; market-neutral funds 0.8–1.2;
Bitcoin buy-and-hold ~0.96 per Fidelity 2020–early-2024). 0.5 sits at the low
end of that honest band — it screens out the breakeven/negative strategies the
first ~10 runs produced (Sharpe −8 to −105) without demanding the improbable."""

MIN_PROFIT_FACTOR_IS: float = 1.2
"""IS profit factor floor (gross wins / gross losses). **Re-tuned 1.5 → 1.2 on
2026-06-08 (SPEC §2/§6).** Directly from the research's recommended bar
("profit factor > 1.2 ... consistent across folds"): 1.2 means gross wins
exceed gross losses by 20%, clearing the ~0.15–0.2% Binance round-trip fee with
margin, without demanding the rich 1.5 that the honest 5m universe rarely
sustains. Matches ``MIN_OOS_PROFIT_FACTOR`` (also 1.2)."""

MAX_DRAWDOWN_IS: float = 0.20
"""Max IS drawdown as positive fraction. 0.20 = 20%. (Unchanged by the
2026-06-08 re-tune — a drawdown ceiling is a risk limit, not a performance
bar.)"""

# ─── Cross-fold consistency (added 2026-06-08, SPEC §2/§6) ──────────────
# The single most important lever the research flagged: a passing AVERAGE is
# not a passing strategy. A high mean Sharpe dragged up by one lucky fold is
# the Deflated-Sharpe selection-bias trap. Consistency across independent
# walk-forward OOS windows is the real signal — so the IS hard gate now
# requires the strategy to make money in a MAJORITY of folds, not on average.

MIN_POSITIVE_FOLDS: int = 4
"""Minimum walk-forward folds (of the BRD §5.4 anchored 6-fold plan) whose
per-fold Sharpe is > 0. **New on 2026-06-08 (SPEC §2/§6).**

The research stressed: "Require consistency across all 6 folds, not a high
average dragged up by one lucky window," and set the advancement bar at
"positive in ≥ 4/6 folds." 4/6 = the strategy made money in two-thirds of the
independent OOS windows — evidence of a repeatable edge rather than a single
fortunate slice that inflates the mean (the Deflated-Sharpe failure mode).
Enforced in ``gate_backtest`` against each fold's realized walk-forward Sharpe
(per-fold ``is_sharpe``; the backtest runs on each fold's OOS test window, so
this IS the out-of-sample per-fold Sharpe — ``oos_sharpe`` stays 0.0 by the
backtest_runner contract). A degraded cache that fits < 4 folds therefore
fails this gate by design: too little evidence to establish consistency."""

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

# ─── Supervisor capacity (BRD §10, §5.1) ─────────────────────────────────
# Two distinct axes the Stage 9 supervisor's spawn gate reasons about. Both
# read from the portfolio snapshot
# (:func:`orchestrator.supervisor.aget_portfolio_snapshot`): ``active`` for
# the resource axis, ``live`` for the capital axis.

MAX_CONCURRENT_STRATEGIES: int = 4
"""Resource axis: max total NON-ARCHIVED threads the supervisor may keep in
flight at once (research / validation / paper_gate / paper / live_gate /
live all count). Bounds the number of Freqtrade containers + LLM-run
threads on the single 8GB/4-core host (BRD §3, §12). The supervisor's
``aspawn_strategy`` refuses a spawn when
``aget_portfolio_snapshot()["active"] >= MAX_CONCURRENT_STRATEGIES`` — no
registry row, no thread kicked. Canonical value table: BRD §10."""

MAX_CONCURRENT_LIVE_STRATEGIES: int = 1
"""Capital axis: max strategies in ``stage="live"`` at once. SPEC §1 Q3
locks $500 live capital and ``live_spawn`` caps each strategy's stake to
the WHOLE ``LIVE_CAPITAL_CAP_USD`` (not a per-strategy slice), so a second
concurrent live strategy would double real exposure beyond the budget —
hence 1 for v1 (option (i), operator fork-A sign-off). This is a
TRANSITION-time constraint (enforced where a paper-graduate is admitted to
live, reading ``aget_portfolio_snapshot()["live"]``), NOT a spawn-time one:
research-spawning is gated only by ``MAX_CONCURRENT_STRATEGIES`` so the
pipeline can keep researching/papering candidates while one runs live.
Raising N is a one-line edit here, but only after SPEC §4.2's capital-ramp
criteria AND a stake-subdivision design (today's whole-amount cap would
over-allocate). Canonical value table: BRD §10."""
