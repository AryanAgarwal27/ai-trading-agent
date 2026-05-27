"""FreqAI 3-class predictor template — LightGBM under the hood (BRD §8.1).

Hypothesis: a supervised classifier trained on technical features can
discriminate short-horizon forward returns into ``{up, flat, down}`` with
enough precision that "trade only on 'up' above ``min_class_prob``" beats
the mean-reversion baseline after fees on the same walk-forward folds.

Template contract (BRD §8):
  - Structural shell is hand-written and untouchable: class name,
    ``feature_engineering_*`` hooks, ``set_freqai_targets``,
    ``populate_indicators`` / ``populate_entry_trend`` / ``populate_exit_trend``,
    ``timeframe``, ``process_only_new_candles``, ``startup_candle_count``.
    The LLM does not edit these.
  - Slots are marked with ``# SLOT: <name> (type, range)`` inline comments
    and are the ONLY values the generator (BRD §5.3) may substitute.
  - Slot names + ranges match ``freqai_classifier_template_schema.py``
    exactly; any divergence is a contract break caught by
    ``tests/unit/test_template_filling.py``.
  - ``freqai_config`` is a hand-written non-slot class attribute carrying
    the BRD §7.3 pins (DI_threshold, SVM outlier rejection,
    train/backtest period, expiration). The orchestrator's config builder
    reads it at paper/live spawn time and merges it into the runtime
    Freqtrade ``config.json``'s ``freqai`` section.

NB. The runtime probability-column name (``&-action_up_proba`` below) is
the conventional FreqAI classifier output for a string target named
``&-action``. If a Freqtrade-version bump changes the convention, the
first paper run is the canary — the column will be missing and the
strategy will silently emit no entries. Stage 7 paper-monitor must alert
on "zero entries with do_predict=1" to surface this.
"""

# ruff: noqa: F401 (freqtrade/talib imports resolved only inside the container)
# pyright: reportMissingImports=false

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import talib.abstract as ta  # type: ignore[import-not-found]

from freqtrade.strategy import IStrategy  # type: ignore[import-not-found]
from freqtrade.vendor.qtpylib import indicators as qtpylib  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from pandas import DataFrame


class FreqaiClassifierTemplate(IStrategy):
    """LightGBM classifier predicts forward direction; entry on high-conviction 'up'.

    Entry (long-only spot): FreqAI confidence (``do_predict==1``) AND predicted
    class 'up' AND its probability ≥ ``min_class_prob`` AND EMA-fast > EMA-slow
    trend filter AND short-term oversold RSI. Exit: predicted class flips off
    'up' OR trend filter inverts.

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

    # `minimal_roi` is required by Freqtrade; we disable ROI-based exits so the
    # strategy's own exit logic (and FreqAI predictions) is the only profit-taker.
    # 12h time-based fallback is a safety net, not a strategy parameter.
    minimal_roi = {"0": 100.0, "720": 0.0}

    # ─── BRD §7.3 FreqAI pins — non-slot, fixed by contract ────────────────
    # The orchestrator config-builder reads this at paper/live spawn time and
    # merges it into the runtime ``config.json``'s ``freqai`` section. These
    # values are deliberately conservative defaults; tuning happens in
    # operator-controlled config overlay, never via SLOT here.
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
    # comment must match a field in FreqaiClassifierParams (schema).

    # RSI lookback (used for the entry oversold filter).
    rsi_period: int = 14  # SLOT: rsi_period (int, 7-30)

    # RSI must be BELOW this to enter — short-term oversold filter on top of
    # the classifier's "up" signal.
    rsi_buy_threshold: int = 25  # SLOT: rsi_buy_threshold (int, 10-40)

    # Fast EMA window (entry trend filter).
    ema_fast: int = 12  # SLOT: ema_fast (int, 5-50)

    # Slow EMA window (entry trend filter). Must be > ema_fast for the cross
    # filter to make sense; schema does NOT enforce ordering — the critic
    # agent (BRD §5.3) checks this in adversarial review.
    ema_slow: int = 50  # SLOT: ema_slow (int, 20-200)

    # Forward window (in candles of `timeframe`) used to compute the training
    # label. Read by ``set_freqai_targets`` below.
    label_period_candles: int = 12  # SLOT: label_period_candles (int, 4-24)

    # |forward_return| > this threshold (in percent) → 'up' or 'down';
    # otherwise 'flat'. Tighter threshold → more 'flat' labels → fewer entries
    # but higher precision per entry.
    label_threshold_pct: float = 0.6  # SLOT: label_threshold_pct (float, 0.1-2.0)

    # Minimum predicted-class probability for the 'up' class to fire an entry.
    # Above 0.7 is "high conviction"; above 0.8 is rarely emitted by a healthy
    # 3-class classifier.
    min_class_prob: float = 0.7  # SLOT: min_class_prob (float, 0.55-0.85)

    # Hard per-trade stoploss (negative fraction).
    stoploss: float = -0.05  # SLOT: stoploss (float, -0.10 to -0.02)

    # ─── FreqAI feature engineering hooks ──────────────────────────────────
    # These three methods are called by FreqAI's training pipeline. Their
    # *names* and *signatures* are fixed by the framework; the LLM does not
    # touch their bodies either — the feature set is part of the template's
    # strategy hypothesis, not a tunable.

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs: object
    ) -> DataFrame:
        """Per-period features — FreqAI iterates ``period`` across the values
        in ``indicator_periods_candles`` in the runtime config."""
        dataframe[f"%-rsi-period_{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"%-mfi-period_{period}"] = ta.MFI(dataframe, timeperiod=period)
        dataframe[f"%-roc-period_{period}"] = ta.ROC(dataframe, timeperiod=period)
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
        dataframe["%-raw_price"] = dataframe["close"]
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
        """Three-class label from forward return over ``label_period_candles``.

        ``label_threshold_pct`` is interpreted as a percent (0.6 → 0.006
        fractional return). ``np.select`` collapses the threshold ladder into a
        single string column ``&-action`` that FreqAI trains the classifier on.
        """
        threshold = self.label_threshold_pct / 100.0
        future_return = (
            dataframe["close"].shift(-self.label_period_candles) / dataframe["close"]
            - 1.0
        )
        conditions = [future_return > threshold, future_return < -threshold]
        choices = ["up", "down"]
        dataframe["&-action"] = np.select(conditions, choices, default="flat")
        return dataframe

    # ─── Indicators (consumed by entry/exit logic, NOT by FreqAI) ──────────

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Compute EMA / RSI columns used by entry/exit, then hand off to FreqAI."""
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period)
        # FreqAI populates ``do_predict``, ``&-action``, and the per-class
        # probability columns by running the trained classifier here.
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    # ─── Entries ───────────────────────────────────────────────────────────

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Long entry: high-conviction 'up' AND uptrend AND short-term oversold."""
        dataframe.loc[
            (
                (dataframe["do_predict"] == 1)
                & (dataframe["&-action"] == "up")
                & (dataframe["&-action_up_proba"] >= self.min_class_prob)
                & (dataframe["ema_fast"] > dataframe["ema_slow"])
                & (dataframe["rsi"] < self.rsi_buy_threshold)
                & (dataframe["volume"] > 0)  # exchange downtime guard
            ),
            "enter_long",
        ] = 1
        return dataframe

    # ─── Exits ─────────────────────────────────────────────────────────────

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Long exit: predicted class falls off 'up' OR trend filter inverts."""
        dataframe.loc[
            (
                (dataframe["&-action"] != "up")
                | (dataframe["ema_fast"] < dataframe["ema_slow"])
            ),
            "exit_long",
        ] = 1
        return dataframe
