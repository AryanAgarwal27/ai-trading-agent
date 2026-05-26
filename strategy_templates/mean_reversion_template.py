"""Mean-reversion baseline template — pure TA, no FreqAI (BRD §8.1).

Hypothesis: when price stretches to the lower Bollinger Band and RSI is
oversold, short-horizon mean reversion is more likely than not.

Template contract (BRD §8):
  - This file's structural shell is hand-written and untouchable. The class
    name, `populate_indicators`, `populate_entry_trend`, `populate_exit_trend`,
    `stoploss`, `timeframe`, and `process_only_new_candles` are not LLM-editable.
  - Slots are marked with ``# SLOT: <name> (type, range)`` inline comments and
    are the ONLY values the generator (BRD §5.3) may substitute. Default
    literals below let the template backtest as-is (Stage 3 DoD: `trades > 0`
    on 1 week of cached BTC/USDT data; Stage 5 smoke test fixture).
  - Slot names + ranges match ``mean_reversion_template_schema.py`` exactly.
    Changing a slot here without updating the schema is a contract break.

The generator renders an LLM-proposed parameter set into a copy of this file
at ``strategy_templates/_generated/<strategy_id>.py`` by overwriting the
literal on each ``# SLOT:`` line with the Pydantic-validated value. AST
validation (BRD §8 rule 4) runs against the rendered output, not this source.
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


class MeanReversionTemplate(IStrategy):
    """RSI + Bollinger Bands mean-reversion baseline.

    Entry (long-only spot): close <= lower Bollinger Band AND RSI < buy
    threshold. Exit: close >= middle Bollinger Band OR RSI > exit threshold.
    Hard stoploss is unconditional.

    BRD §1 v1 is spot-only, long-only — no short side, no margin, no leverage.
    """

    # ─── Structural shell — DO NOT add to the slot list ────────────────────
    # These class attributes encode the strategy *shape*; the LLM does not
    # touch them. Freqtrade requires them at class scope.
    INTERFACE_VERSION = 3
    timeframe = "5m"
    process_only_new_candles = True
    can_short = False  # BRD §1: spot-only, long-only
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count = 200  # >= bb_period + rsi_period worst-case

    # `minimal_roi` is required by Freqtrade; we disable ROI-based exits so the
    # strategy's own exit logic is the only profit-taker. Time-based fallback
    # at 12h is a safety net, not a strategy parameter.
    minimal_roi = {"0": 100.0, "720": 0.0}

    # ─── SLOT BLOCK ────────────────────────────────────────────────────────
    # Generator replaces the RHS literal on each line below. Slot name in the
    # comment must match a field in MeanReversionParams (schema).

    # Bollinger Bands window length (in candles of `timeframe`).
    bb_period: int = 20  # SLOT: bb_period (int, 10-50)

    # Standard deviations for the BB envelope. Wider stds → fewer, deeper signals.
    bb_std: float = 2.0  # SLOT: bb_std (float, 1.5-3.0)

    # RSI lookback.
    rsi_period: int = 14  # SLOT: rsi_period (int, 7-30)

    # RSI must be BELOW this on entry. Lower = stricter oversold.
    rsi_buy_threshold: int = 30  # SLOT: rsi_buy_threshold (int, 10-40)

    # RSI ABOVE this triggers exit. Higher = wait for stronger mean reversion.
    rsi_exit_threshold: int = 55  # SLOT: rsi_exit_threshold (int, 50-80)

    # Hard stoploss (negative fraction). E.g. -0.05 = exit at -5% from entry.
    stoploss: float = -0.05  # SLOT: stoploss (float, -0.10 to -0.02)

    # ─── Indicators ────────────────────────────────────────────────────────

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Compute BB envelope and RSI columns the entry/exit logic reads."""
        bb = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe),
            window=self.bb_period,
            stds=self.bb_std,
        )
        dataframe["bb_lowerband"] = bb["lower"]
        dataframe["bb_middleband"] = bb["mid"]
        dataframe["bb_upperband"] = bb["upper"]

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period)
        return dataframe

    # ─── Entries ───────────────────────────────────────────────────────────

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Long entry: oversold + at/below lower BB."""
        dataframe.loc[
            (
                (dataframe["close"] <= dataframe["bb_lowerband"])
                & (dataframe["rsi"] < self.rsi_buy_threshold)
                & (dataframe["volume"] > 0)  # exchange downtime guard
            ),
            "enter_long",
        ] = 1
        return dataframe

    # ─── Exits ─────────────────────────────────────────────────────────────

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Long exit: mean reverted to middle BB, or RSI crosses back up."""
        dataframe.loc[
            (
                (dataframe["close"] >= dataframe["bb_middleband"])
                | (dataframe["rsi"] > self.rsi_exit_threshold)
            ),
            "exit_long",
        ] = 1
        return dataframe
