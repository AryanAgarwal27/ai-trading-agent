"""Regime-filtered Bollinger mean-reversion template — pure TA, no FreqAI.

Hypothesis: in a confirmed uptrend (price above a long EMA), short, sharp dips
that stretch BELOW the lower Bollinger Band and print an oversold RSI tend to
revert toward the band's mean. Restricting entries to the uptrend regime is the
load-bearing idea: it keeps the strategy OUT of falling markets (where buying
dips is catching knives) and only buys pullbacks inside an upward drift. Exit is
fast — a small fixed take-profit OR reversion to the band mean — so capital is
recycled quickly and losers are cut by a hard stop.

This archetype was selected because, on the project's own anchored 6-fold
walk-forward over the cached BTC/ETH/SOL/BNB data, it is the only long-only shape
that produced a positive, consistent risk-adjusted return through a bearish OOS
window (it sits out the downtrends via the EMA regime filter). See the strategy
README for the proxy-backtest evidence; the authoritative numbers come from
running this through POST /strategies/validate.

Template contract (BRD §8):
  - The structural shell (class name, populate_* methods, timeframe, stoploss,
    process_only_new_candles, protections) is hand-written and NOT LLM-editable.
  - Slots are marked ``# SLOT: <name> (type, range)`` and are the ONLY values the
    generator may substitute. Default literals let the template backtest as-is.
  - Slot names + ranges match ``bb_regime_reversion_template_schema.py`` exactly.

BRD §1 v1 is spot-only, long-only — no short side, no margin, no leverage.
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


class BbRegimeReversionTemplate(IStrategy):
    """Regime-filtered Bollinger + RSI mean-reversion.

    Entry (long-only spot): close <= lower Bollinger Band AND RSI < buy threshold
    AND close > trend EMA (uptrend regime). Exit: close >= middle Bollinger Band,
    OR a fixed take-profit (``minimal_roi``), OR the hard stoploss.
    """

    # ─── Structural shell — DO NOT add to the slot list ────────────────────
    INTERFACE_VERSION = 3
    timeframe = "15m"
    process_only_new_candles = True
    can_short = False  # BRD §1: spot-only, long-only
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count = 450  # >= ema_trend_period(<=400) + bb_period worst-case

    # ─── SLOT BLOCK ────────────────────────────────────────────────────────
    # Generator replaces the RHS literal on each line below. Slot name in the
    # comment must match a field in BbRegimeReversionParams (schema).

    # Bollinger Bands window length (in candles of `timeframe`).
    bb_period: int = 20  # SLOT: bb_period (int, 10-50)

    # Standard deviations for the BB envelope. Wider → deeper, rarer dips.
    bb_std: float = 2.3  # SLOT: bb_std (float, 1.5-3.0)

    # RSI lookback.
    rsi_period: int = 14  # SLOT: rsi_period (int, 7-30)

    # RSI must be BELOW this on entry. Lower = stricter oversold (fewer, higher-quality).
    rsi_buy_threshold: int = 35  # SLOT: rsi_buy_threshold (int, 10-45)

    # Trend regime filter: only buy dips when close is ABOVE this EMA.
    ema_trend_period: int = 200  # SLOT: ema_trend_period (int, 50-400)

    # Fixed take-profit fraction (e.g. 0.025 = +2.5%). Wired into minimal_roi below.
    roi_target: float = 0.025  # SLOT: roi_target (float, 0.005-0.05)

    # Hard stoploss (negative fraction). E.g. -0.06 = exit at -6% from entry.
    stoploss: float = -0.06  # SLOT: stoploss (float, -0.10 to -0.02)

    # `minimal_roi` is built from the roi_target slot: a single flat take-profit at
    # +roi_target from entry (minute 0). Reverts-to-mean exits are handled by the
    # exit signal below; the hard stoploss caps losers. Referencing the slot here
    # keeps the take-profit in sync with whatever value the generator renders.
    minimal_roi = {"0": roi_target}

    # ─── In-container risk protection — structural shell, NOT a slot ───────
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
        """Long entry: oversold dip below lower BB, but only in an uptrend regime."""
        dataframe.loc[
            (
                (dataframe["close"] <= dataframe["bb_lowerband"])
                & (dataframe["rsi"] < self.rsi_buy_threshold)
                & (dataframe["close"] > dataframe["ema_trend"])
                & (dataframe["volume"] > 0)  # exchange downtime guard
            ),
            "enter_long",
        ] = 1
        return dataframe

    # ─── Exits ─────────────────────────────────────────────────────────────

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Long exit: price reverted up to the middle band."""
        dataframe.loc[
            (dataframe["close"] >= dataframe["bb_middleband"]),
            "exit_long",
        ] = 1
        return dataframe
