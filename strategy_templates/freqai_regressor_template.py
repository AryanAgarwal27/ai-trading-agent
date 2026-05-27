"""FreqAI regression template — LightGBM predicts forward return (BRD §8.1).

Hypothesis: a supervised regressor trained on technical features can
estimate the *magnitude* of the next ``label_period_candles`` return
well enough that "enter when predicted_return > k * ATR%" produces a
positive-expectancy long-only spot strategy after fees.

Template contract (BRD §8):
  - Structural shell is hand-written and untouchable: class name,
    ``feature_engineering_*`` hooks, ``set_freqai_targets``,
    ``populate_indicators`` / ``populate_entry_trend`` / ``populate_exit_trend``,
    ``timeframe``, ``process_only_new_candles``, ``startup_candle_count``.
    The LLM does not edit these.
  - Slots are marked with ``# SLOT: <name> (type, range)`` inline comments
    and are the ONLY values the generator (BRD §5.3) may substitute.
  - Slot names + ranges match ``freqai_regressor_template_schema.py``
    exactly; any divergence is a contract break caught by
    ``tests/unit/test_template_filling.py``.
  - ``freqai_config`` is a hand-written non-slot class attribute carrying
    the BRD §7.3 pins. The orchestrator's config builder reads it at
    paper/live spawn time and merges it into the runtime Freqtrade
    ``config.json``'s ``freqai`` section.

The regressor target ``&-future_return`` is a *fractional* return. The
entry compares it to ``k_atr_multiplier * (ATR / close)`` — both sides
are dimensionless fractions, which keeps the threshold meaningful across
volatility regimes (a 0.5% predicted return is interesting in a calm
regime but unremarkable in a vol-of-vol spike).
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


class FreqaiRegressorTemplate(IStrategy):
    """LightGBM regressor predicts forward return; entry when prediction beats ATR.

    Entry (long-only spot): FreqAI confidence (``do_predict==1``) AND predicted
    forward fractional return > ``k_atr_multiplier * (ATR/close)`` AND
    EMA-fast > EMA-slow trend filter. Exit: prediction turns negative OR
    trend filter inverts.

    BRD §1 v1 is spot-only, long-only — no short side, margin, or leverage.
    """

    # ─── Structural shell — DO NOT add to the slot list ────────────────────
    INTERFACE_VERSION = 3
    timeframe = "5m"
    process_only_new_candles = True
    can_short = False  # BRD §1: spot-only, long-only
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count = 200  # >= longest EMA + FreqAI lookback worst-case

    minimal_roi = {"0": 100.0, "720": 0.0}

    # ─── BRD §7.3 FreqAI pins — non-slot, fixed by contract ────────────────
    freqai_config = {
        "train_period_days": 30,
        "backtest_period_days": 7,
        "live_retrain_hours": 24,
        "expiration_hours": 72,
        "purge_old_models": 2,
        "feature_parameters": {
            "DI_threshold": 0.9,
            "use_SVM_to_remove_outliers": True,
        },
    }

    # ─── SLOT BLOCK ────────────────────────────────────────────────────────
    # Generator replaces the RHS literal on each line below. Slot name in the
    # comment must match a field in FreqaiRegressorParams (schema).

    # Fast EMA window (entry trend filter).
    ema_fast: int = 12  # SLOT: ema_fast (int, 5-50)

    # Slow EMA window (entry trend filter). Should be > ema_fast for the cross
    # filter to make sense; schema doesn't enforce ordering — critic does.
    ema_slow: int = 50  # SLOT: ema_slow (int, 20-200)

    # ATR lookback (used to compute the dynamic entry threshold).
    atr_period: int = 14  # SLOT: atr_period (int, 7-30)

    # Forward window (in candles of `timeframe`) used to compute the training
    # label. Read by ``set_freqai_targets`` below.
    label_period_candles: int = 12  # SLOT: label_period_candles (int, 4-24)

    # Entry threshold multiplier on ATR%: enter when
    #   predicted_return > k_atr_multiplier * (ATR / close).
    # Higher k → fewer, higher-conviction entries.
    k_atr_multiplier: float = 1.5  # SLOT: k_atr_multiplier (float, 0.5-3.0)

    # Hard per-trade stoploss (negative fraction).
    stoploss: float = -0.05  # SLOT: stoploss (float, -0.10 to -0.02)

    # ─── FreqAI feature engineering hooks ──────────────────────────────────

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs: object
    ) -> DataFrame:
        """Per-period features — FreqAI iterates ``period`` across the values
        in ``indicator_periods_candles`` in the runtime config."""
        dataframe[f"%-rsi-period_{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"%-roc-period_{period}"] = ta.ROC(dataframe, timeperiod=period)
        dataframe[f"%-atr-period_{period}"] = ta.ATR(dataframe, timeperiod=period)
        bb = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe[f"%-bb_width_{period}"] = (bb["upper"] - bb["lower"]) / bb["mid"]
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs: object
    ) -> DataFrame:
        """Period-independent basic features."""
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs: object
    ) -> DataFrame:
        """Calendar features (always added once per dataframe)."""
        dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        dataframe["%-hour_of_day"] = (dataframe["date"].dt.hour + 1) / 24
        return dataframe

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs: object
    ) -> DataFrame:
        """Regression target: fractional forward return over ``label_period_candles``.

        FreqAI strips the trailing ``label_period_candles`` rows from the
        training window (their forward look would peek past the end), so this
        is a legitimate label rather than a leak.
        """
        dataframe["&-future_return"] = (
            dataframe["close"].shift(-self.label_period_candles) / dataframe["close"]
            - 1.0
        )
        return dataframe

    # ─── Indicators (consumed by entry/exit logic, NOT by FreqAI) ──────────

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Compute EMA / ATR columns used by entry/exit, then hand off to FreqAI."""
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period)
        # FreqAI populates ``do_predict`` and ``&-future_return`` here.
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    # ─── Entries ───────────────────────────────────────────────────────────

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Long entry: predicted return beats ATR-scaled threshold AND uptrend."""
        # Normalize ATR to a fractional move so it is comparable to the
        # regressor's fractional-return prediction.
        atr_pct = dataframe["atr"] / dataframe["close"]
        dataframe.loc[
            (
                (dataframe["do_predict"] == 1)
                & (dataframe["&-future_return"] > self.k_atr_multiplier * atr_pct)
                & (dataframe["ema_fast"] > dataframe["ema_slow"])
                & (dataframe["volume"] > 0)  # exchange downtime guard
            ),
            "enter_long",
        ] = 1
        return dataframe

    # ─── Exits ─────────────────────────────────────────────────────────────

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Long exit: predicted return turns negative OR trend filter inverts."""
        dataframe.loc[
            (
                (dataframe["&-future_return"] < 0)
                | (dataframe["ema_fast"] < dataframe["ema_slow"])
            ),
            "exit_long",
        ] = 1
        return dataframe
