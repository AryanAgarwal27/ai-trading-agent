"""Regime-filtered Donchian breakout template — pure TA, no FreqAI.

Hypothesis: when a confirmed uptrend is in force (a fast EMA above a slow EMA),
price breaking above its recent N-candle high signals trend continuation. Ride
the move with a trailing stop so winners run and losers are cut early; abandon
the position when the trend regime flips (fast EMA crosses back below slow).

This is the COMPLEMENTARY archetype to the mean-reversion template: it is built
to harvest sustained directional moves, which a mean-reversion strategy gives up.
NOTE ON REGIME: trend-following needs trends. On the project's current anchored
walk-forward window (a bearish/choppy OOS stretch) this strategy does NOT clear
the gate — long breakouts get chopped up when there is no sustained uptrend. It
is included so the portfolio has a strategy ready to promote when the anchored
window contains an up-trending regime (re-run validation after the OHLCV cache
advances into a bull phase). Always confirm via POST /strategies/validate.

Template contract (BRD §8):
  - The structural shell is hand-written and NOT LLM-editable.
  - Slots are marked ``# SLOT: <name> (type, range)`` and are the ONLY values the
    generator may substitute.
  - Slot names + ranges match ``donchian_regime_trend_template_schema.py`` exactly.

BRD §1 v1 is spot-only, long-only — no short side, no margin, no leverage.
"""

# ruff: noqa: F401 (freqtrade/talib imports resolved only inside the container)
# pyright: reportMissingImports=false

from __future__ import annotations

from typing import TYPE_CHECKING

import talib.abstract as ta  # type: ignore[import-not-found]

from freqtrade.strategy import IStrategy  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from pandas import DataFrame


class DonchianRegimeTrendTemplate(IStrategy):
    """Regime-filtered Donchian breakout with a trailing-stop exit.

    Entry (long-only spot): close > prior ``donchian_period``-candle high AND
    fast EMA > slow EMA (uptrend regime). Exit: fast EMA < slow EMA (regime flip),
    a trailing stop, or the hard stoploss. ROI exits are disabled so winners run.
    """

    # ─── Structural shell — DO NOT add to the slot list ────────────────────
    INTERFACE_VERSION = 3
    timeframe = "1h"
    process_only_new_candles = True
    can_short = False  # BRD §1: spot-only, long-only
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count = 300  # >= ema_slow(<=200) + donchian_period worst-case

    # ROI exits disabled: a trend strategy must let winners run. The trailing
    # stop + regime-flip exit + hard stop manage exits. ``100`` = +10000%, never hit.
    minimal_roi = {"0": 100.0}

    # Trailing stop is the primary winner-protection mechanism. ``trailing_stop``
    # is structural (always on for this archetype); the DISTANCE is the slot below.
    trailing_stop = True
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    # ─── SLOT BLOCK ────────────────────────────────────────────────────────
    # Generator replaces the RHS literal on each line below. Slot name in the
    # comment must match a field in DonchianRegimeTrendParams (schema).

    # Donchian lookback: breakout above the highest high of the prior N candles.
    donchian_period: int = 20  # SLOT: donchian_period (int, 10-100)

    # Fast EMA of the regime filter.
    ema_fast: int = 20  # SLOT: ema_fast (int, 10-100)

    # Slow EMA of the regime filter. Uptrend regime = ema_fast > ema_slow.
    ema_slow: int = 50  # SLOT: ema_slow (int, 30-200)

    # Trailing-stop distance (positive fraction). E.g. 0.05 = exit 5% below peak.
    trailing_stop_positive: float = 0.05  # SLOT: trailing_stop_positive (float, 0.02-0.15)

    # Hard stoploss (negative fraction). Backstop below the trailing stop.
    stoploss: float = -0.10  # SLOT: stoploss (float, -0.15 to -0.03)

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
        """Compute the Donchian upper channel (prior-N high) and regime EMAs."""
        # `.shift(1)` excludes the current candle so the breakout test compares
        # against the PRIOR window's high — no look-ahead.
        dataframe["donchian_high"] = dataframe["high"].rolling(self.donchian_period).max().shift(1)
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow)
        return dataframe

    # ─── Entries ───────────────────────────────────────────────────────────

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Long entry: breakout above the prior-N high, in an uptrend regime."""
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["donchian_high"])
                & (dataframe["ema_fast"] > dataframe["ema_slow"])
                & (dataframe["volume"] > 0)  # exchange downtime guard
            ),
            "enter_long",
        ] = 1
        return dataframe

    # ─── Exits ─────────────────────────────────────────────────────────────

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Long exit: the uptrend regime flipped off (fast EMA below slow EMA)."""
        dataframe.loc[
            (dataframe["ema_fast"] < dataframe["ema_slow"]),
            "exit_long",
        ] = 1
        return dataframe
