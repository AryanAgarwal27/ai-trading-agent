"""Regime-filtered Bollinger SHORT template — pure TA, no FreqAI.

The mirror image of ``bb_regime_reversion_template`` (which longs oversold dips
in an uptrend). Hypothesis: in a confirmed DOWNtrend (price below a long EMA),
sharp rallies that stretch ABOVE the upper Bollinger Band and print an overbought
RSI tend to fade back toward the band's mean — so SHORT the rally and cover at the
mean. Restricting entries to the downtrend regime is the load-bearing idea: it
keeps the strategy OUT of rising markets (where shorting strength is fighting the
trend) and only shorts exhaustion rallies inside a downward drift. Exit (cover) is
fast — a small fixed take-profit OR reversion to the band mean — so capital is
recycled quickly and losers are cut by a hard stop.

WHY THIS ARCHETYPE (Stage 13 Phase 1, BRD §22.1): the project's validation window
is a bear market (Nov 2025–May 2026; every SPEC §1 Q2 pair fell 12–39%). A
long-only spot system structurally cannot profit from that dominant downward move;
the short side can. This template lets the BACKTEST evaluate whether shorting has
edge against the SAME BRD §10 gates as a long strategy — with ZERO live exposure.

SHORT CAPABILITY + EXECUTION SCOPE (BRD §22.1):
  - ``can_short = True`` and ``populate_*`` emit Freqtrade's native
    ``enter_short`` / ``exit_short`` columns. A short backtest requires the
    runtime config to run in futures trading mode; the backtest runner
    (``orchestrator/tools/backtest_runner.py``) detects ``can_short = True`` and
    renders ``trading_mode="futures"`` + ``margin_mode="isolated"`` for this
    strategy ONLY (the spot/long path stays byte-identical). Leverage stays at
    the Freqtrade 1× default — this template defines NO ``leverage()`` callback;
    leverage is a Phase-2 (live) concern, not a Phase-1 (backtest-signal) one.
  - **Phase 1 is backtest/validation ONLY.** A short-capable strategy that
    passes the full gauntlet is deliberately BLOCKED from promotion to the
    spot paper/live path at ``paper_spawn`` (status
    ``short_paper_deferred_to_phase2``). It can be VALIDATED but cannot trade
    real money until Stage 13 Phase 2 ships the live futures/margin risk model
    (BRD §22.2). Nothing in this template touches a live account.

Template contract (BRD §8):
  - The structural shell (class name, populate_* methods, timeframe, stoploss,
    process_only_new_candles, protections, can_short) is hand-written and NOT
    LLM-editable.
  - Slots are marked ``# SLOT: <name> (type, range)`` and are the ONLY values the
    generator may substitute. Default literals let the template backtest as-is.
  - Slot names + ranges match ``bb_regime_short_template_schema.py`` exactly.
"""

# ruff: noqa: F401 (freqtrade/talib imports resolved only inside the container)
# pyright: reportMissingImports=false

from __future__ import annotations

from typing import TYPE_CHECKING

import talib.abstract as ta  # type: ignore[import-not-found]

from freqtrade.strategy import IStrategy  # type: ignore[import-not-found]
from freqtrade.vendor.qtpylib import indicators as qtpylib  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from pandas import DataFrame


class BbRegimeShortTemplate(IStrategy):
    """Regime-filtered Bollinger + RSI SHORT (the short-side mirror of bb_regime_reversion).

    Entry (short): close >= upper Bollinger Band AND RSI > sell threshold AND
    close < trend EMA (downtrend regime). Exit (cover): close <= middle Bollinger
    Band, OR a fixed take-profit (``minimal_roi``), OR the hard stoploss.

    Stage 13 Phase 1 (BRD §22.1): backtest/validation only. ``can_short = True``
    requires futures trading mode in the backtest config; the live/paper SPOT
    path stays untouched and this strategy is blocked from spot-paper promotion.
    """

    # ─── Structural shell — DO NOT add to the slot list ────────────────────
    INTERFACE_VERSION = 3
    timeframe = "15m"
    process_only_new_candles = True
    can_short = True  # BRD §22.1: short signals evaluated in a FUTURES backtest only
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count = 450  # >= ema_trend_period(<=400) + bb_period worst-case

    # ─── SLOT BLOCK ────────────────────────────────────────────────────────
    # Generator replaces the RHS literal on each line below. Slot name in the
    # comment must match a field in BbRegimeShortParams (schema).

    # Bollinger Bands window length (in candles of `timeframe`).
    bb_period: int = 20  # SLOT: bb_period (int, 10-50)

    # Standard deviations for the BB envelope. Wider → rarer, more-stretched rallies.
    bb_std: float = 2.3  # SLOT: bb_std (float, 1.5-3.0)

    # RSI lookback.
    rsi_period: int = 14  # SLOT: rsi_period (int, 7-30)

    # RSI must be ABOVE this on entry. Higher = stricter overbought (fewer, higher-quality shorts).
    rsi_sell_threshold: int = 70  # SLOT: rsi_sell_threshold (int, 55-90)

    # Trend regime filter: only short rallies when close is BELOW this EMA (downtrend).
    ema_trend_period: int = 200  # SLOT: ema_trend_period (int, 50-400)

    # Fixed take-profit fraction (e.g. 0.025 = +2.5% gain on the short). Wired into minimal_roi.
    roi_target: float = 0.025  # SLOT: roi_target (float, 0.005-0.05)

    # Hard stoploss (negative fraction). For a short, the adverse move is price
    # RISING; e.g. -0.06 = cover at a 6% loss against the position.
    stoploss: float = -0.06  # SLOT: stoploss (float, -0.10 to -0.02)

    # `minimal_roi` is built from the roi_target slot: a single flat take-profit at
    # +roi_target (minute 0). For a short, ROI realises when price falls. Reverts-
    # to-mean exits are handled by the exit signal below; the hard stoploss caps
    # losers. Referencing the slot keeps the take-profit in sync with the render.
    minimal_roi = {"0": roi_target}

    # ─── In-container risk protection — structural shell, NOT a slot ───────
    # Identical MaxDrawdown net as the other templates (BRD §11 in-container
    # safety; config-level `protections` is deprecated in Freqtrade 2026.4, so it
    # lives on the strategy class). The out-of-band kill switch stays the
    # authoritative enforcer. The LLM never edits this (BRD §8 rule 1).
    @property
    def protections(self) -> list[dict]:
        return [
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 288,
                "trade_limit": 4,
                "stop_duration_candles": 12,
                "max_allowed_drawdown": 0.12,
            },
        ]

    # ─── Indicators ────────────────────────────────────────────────────────

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Compute BB envelope, RSI, and the trend EMA the logic reads."""
        bb = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe),
            window=self.bb_period,
            stds=self.bb_std,
        )
        dataframe["bb_lowerband"] = bb["lower"]
        dataframe["bb_middleband"] = bb["mid"]
        dataframe["bb_upperband"] = bb["upper"]

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period)
        dataframe["ema_trend"] = ta.EMA(dataframe, timeperiod=self.ema_trend_period)
        return dataframe

    # ─── Entries ───────────────────────────────────────────────────────────

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Short entry: overbought rally above the upper BB, but only in a downtrend regime."""
        dataframe.loc[
            (
                (dataframe["close"] >= dataframe["bb_upperband"])
                & (dataframe["rsi"] > self.rsi_sell_threshold)
                & (dataframe["close"] < dataframe["ema_trend"])
                & (dataframe["volume"] > 0)  # exchange downtime guard
            ),
            "enter_short",
        ] = 1
        return dataframe

    # ─── Exits ─────────────────────────────────────────────────────────────

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Short exit (cover): price reverted down to the middle band."""
        dataframe.loc[
            (dataframe["close"] <= dataframe["bb_middleband"]),
            "exit_short",
        ] = 1
        return dataframe
