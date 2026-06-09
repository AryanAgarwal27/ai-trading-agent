"""FreqAI triple-barrier classifier template — LightGBM under the hood (BRD §8.1).

Research-designed replacement for the forward-return regressor (which scored
Sharpe −24 to −105). The diagnosis (attached research, §A): a raw forward-return
target at 5m is noise-dominated, a featureless input set makes the model predict
near the unconditional mean, and fee drag eats the per-trade edge. This template
attacks all three legs:

  1. **Target (§B):** a *volatility-scaled triple-barrier* CLASSIFICATION label
     (López de Prado, AFML ch. 3) instead of a raw forward return — each bar is
     labelled by which barrier it touches first (vol-scaled profit-take vs stop
     within a ``label_period_candles`` horizon), modelled by a LightGBM
     **classifier**. Path-dependent, vol-normalised, learnable.
  2. **Features (§C):** a compact ~50–80-column set grounded in documented crypto
     effects — time-series momentum (RSI/ROC/ADX/dist-EMA), mean-reversion
     (z-score, Bollinger %B), volatility (ATR%, realised vol), and
     volume/microstructure (vol-z, MFI, OBV, VWAP-distance) — across the
     ``include_timeframes`` (5m + 1h) and BTC/ETH correlated pairs.
  3. **Entry gate (§D):** a meta-labelling gate — act only when
     ``do_predict==1`` AND ``P(up) > entry_proba`` AND ADX regime is trending
     AND price is above its EMA200 trend filter. Exit on ATR target / ATR stop
     (custom callbacks) or when the model flips bearish.

Template contract (BRD §8):
  - Structural shell is hand-written and untouchable: class name,
    ``feature_engineering_*`` hooks, ``set_freqai_targets``,
    ``populate_indicators`` / ``populate_entry_trend`` / ``populate_exit_trend``,
    ``custom_stoploss`` / ``custom_exit``, ``timeframe``,
    ``process_only_new_candles``, ``startup_candle_count``. The LLM never edits
    these.
  - Slots are marked with ``# SLOT: <name> (type, range)`` inline comments and
    are the ONLY values the generator (BRD §5.3) / manual-inject (BRD §21.2) may
    substitute. Slot names + ranges match
    ``freqai_triple_barrier_template_schema.py`` byte-for-byte — drift is caught
    by ``tests/unit/test_template_filling.py``.
  - ``freqai_config`` is a hand-written non-slot class attribute carrying the
    BRD §7.3 pins + the feature SCOPE (include_timeframes / corr pairs /
    indicator periods). The ELEVEN model/FreqAI tunables (``n_estimators`` …
    ``bagging_fraction``, ``DI_threshold``, ``weight_factor``) are SLOTS — class
    attributes — and ``orchestrator.tools.freqai_config.build_freqai_config``
    reads them off the (rendered) class at backtest time and assembles them into
    the runtime ``model_training_parameters`` / ``feature_parameters``. So the
    SLOTS are the single source of truth for the tunable subset and the runtime
    config Freqtrade sees still carries the full research block — no drift
    between a class-attribute slot and a literal buried in a config dict.

**Long-only spot (BRD §1):** ``can_short = False``; no margin, no leverage.

**Timeframe (1h-capable, not 5m-hardcoded):** the class default is ``"5m"`` but
Freqtrade's config ``timeframe`` overrides it, and ``build_freqai_config`` filters
``include_timeframes`` to the base timeframe + strictly-higher TFs — so the same
template runs at ``"1h"`` (research §E recommends testing 1h, where the per-trade
edge survives fees better) by passing ``timeframe="1h"`` to the backtest/spawn.

NB (research §D, "confirm the probability column name for your FreqAI version" —
and we DID, against freqtrade 2026.4's stable_freqai source): for a string target
``&-trade`` ∈ {"up","down"}, ``BaseClassifierModel.predict`` builds the
probability columns with ``columns=self.model.classes_`` — i.e. named after the
CLASS VALUES themselves — and ``DataKitchen.get_predictions_to_append`` appends
them UNPREFIXED. So P(up) is the plain column ``"up"`` (and ``"down"`` is
P(down)); the predicted-label column is ``&-trade``. This is the research's
original convention (``df["up"]``), confirmed by the Stage 12 freqtrade
integration test — which initially FAILED on a wrong ``&-trade_up_proba`` guess
(KeyError in ``populate_entry_trend``), proving the test is a real version-rename
canary, not a rubber stamp.
"""

# ruff: noqa: F401 (freqtrade/talib imports resolved only inside the container)
# pyright: reportMissingImports=false

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import talib.abstract as ta  # type: ignore[import-not-found]

from freqtrade.strategy import IStrategy  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from pandas import DataFrame


class FreqaiTripleBarrierClassifier(IStrategy):
    """Triple-barrier LightGBM classifier; entry on high-conviction P(up) in a trend.

    Entry (long-only spot): ``do_predict==1`` AND ``P(up) > entry_proba`` AND
    ``ADX > adx_min`` (trending regime) AND ``close > EMA200`` (long-only trend
    filter). Exit: model flips bearish (``P(up) < 0.5``); ATR target / ATR stop
    governed by ``custom_exit`` / ``custom_stoploss``.

    The class name contains "Classifier" so
    ``orchestrator.tools.freqai_config.freqai_model_for`` resolves
    ``--freqaimodel LightGBMClassifier`` (it keys off the substring) — the
    classifier path, not the regressor default.
    """

    # ─── Structural shell — DO NOT add to the slot list ────────────────────
    INTERFACE_VERSION = 3
    timeframe = "5m"  # config-overridable; build_freqai_config makes 1h work too
    process_only_new_candles = True
    can_short = False  # BRD §1: spot-only, long-only
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    use_custom_stoploss = True  # REQUIRED for custom_stoploss() to be called
    # EMA200 + vol/label windows are the lookback floor; 300 covers the worst case.
    startup_candle_count = 300

    # ROI disabled — the ATR target (custom_exit) + model exit are the
    # profit-takers; the 12h fallback is a safety net, not a strategy parameter.
    minimal_roi = {"0": 100.0, "720": 0.0}

    # Hard per-trade backstop (negative fraction). The ATR custom_stoploss is the
    # primary stop; this is the floor Freqtrade requires. NOT a slot.
    stoploss = -0.05

    # ─── BRD §7.3 FreqAI pins + feature scope — non-slot, fixed by contract ─
    # Pure literal (ast.literal_eval'd by extract_freqai_pins). Carries the §7.3
    # pins and the feature SCOPE; the tunable model params + DI_threshold +
    # weight_factor are SLOTS below and build_freqai_config reads them off the
    # class — see module docstring.
    freqai_config = {
        "train_period_days": 30,
        "backtest_period_days": 7,
        "live_retrain_hours": 24,
        "expiration_hours": 72,
        "purge_old_models": 2,
        "feature_parameters": {
            "use_SVM_to_remove_outliers": True,
            "include_timeframes": ["5m", "1h"],
            "include_corr_pairlist": ["BTC/USDT", "ETH/USDT"],
            "include_shifted_candles": 2,
            "indicator_periods_candles": [14, 48],
        },
    }

    # ─── SLOT BLOCK ────────────────────────────────────────────────────────
    # Generator / manual-inject replaces the RHS literal on each line below. The
    # slot name in the comment must match a field in FreqaiTripleBarrierParams.
    # Defaults + ranges are research §D2.

    # Strategy-logic slots (read by the strategy methods via self.<name>).

    # Minimum P(up) from the classifier to fire an entry (meta-gate threshold).
    entry_proba: float = 0.60  # SLOT: entry_proba (float, 0.50-0.80)

    # ADX floor — only trade when the regime is trending (gates out chop).
    adx_min: int = 20  # SLOT: adx_min (int, 15-35)

    # Triple-barrier PROFIT-take width, in units of recent realised vol.
    tb_profit_mult: float = 2.0  # SLOT: tb_profit_mult (float, 1.0-4.0)

    # Triple-barrier STOP width, in units of recent realised vol.
    tb_stop_mult: float = 1.5  # SLOT: tb_stop_mult (float, 1.0-4.0)

    # Rolling window (candles) for the realised-vol used to scale the barriers.
    tb_vol_window: int = 48  # SLOT: tb_vol_window (int, 20-100)

    # ATR-target multiple for custom_exit (profit-take in ATR units).
    atr_target_mult: float = 2.0  # SLOT: atr_target_mult (float, 1.0-4.0)

    # ATR-stop multiple for custom_stoploss (stop in ATR units).
    atr_stop_mult: float = 1.5  # SLOT: atr_stop_mult (float, 1.0-3.0)

    # Vertical-barrier horizon AND FreqAI label window (candles of `timeframe`).
    # Research §B: hours, not minutes — 24-72 candles at 5m = 2-6h.
    label_period_candles: int = 36  # SLOT: label_period_candles (int, 24-72)

    # Model/FreqAI slots — read OFF THE CLASS by build_freqai_config and assembled
    # into the runtime model_training_parameters / feature_parameters (see
    # module docstring). They are class attributes purely so they are fillable
    # slots; the strategy methods do not read them.

    n_estimators: int = 200  # SLOT: n_estimators (int, 100-400)
    learning_rate: float = 0.02  # SLOT: learning_rate (float, 0.005-0.05)
    num_leaves: int = 16  # SLOT: num_leaves (int, 8-31)
    max_depth: int = 4  # SLOT: max_depth (int, 3-6)
    min_child_samples: int = 80  # SLOT: min_child_samples (int, 40-150)
    lambda_l1: float = 1.0  # SLOT: lambda_l1 (float, 0.0-5.0)
    lambda_l2: float = 1.0  # SLOT: lambda_l2 (float, 0.0-5.0)
    feature_fraction: float = 0.7  # SLOT: feature_fraction (float, 0.5-0.9)
    bagging_fraction: float = 0.7  # SLOT: bagging_fraction (float, 0.5-0.9)
    DI_threshold: float = 0.9  # SLOT: DI_threshold (float, 0.0-2.0)
    weight_factor: float = 0.5  # SLOT: weight_factor (float, 0.0-1.0)

    # ─── In-container risk protection — structural shell, NOT a slot ───────
    # Freqtrade 2026.4 deprecated config-level ``protections`` (the stable_freqai
    # image REJECTS it at boot); a strategy ``protections`` @property is the only
    # supported location. Identical MaxDrawdown net as the other shipped
    # templates (BRD §11 in-container safety; the out-of-band APScheduler kill
    # switch stays authoritative). The LLM never edits this (BRD §8 rule 1).
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

    # ─── FreqAI feature engineering hooks (research §C) ────────────────────
    # Names + signatures are fixed by the framework; the LLM does not touch the
    # bodies — the feature set IS the strategy hypothesis, not a tunable.

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs: object
    ) -> DataFrame:
        """Per-period features — FreqAI iterates ``period`` across
        ``indicator_periods_candles`` in the runtime config."""
        # MOMENTUM / TREND — time-series momentum (Liu & Tsyvinski).
        dataframe[f"%-rsi-period_{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"%-roc-period_{period}"] = ta.ROC(dataframe, timeperiod=period)
        dataframe[f"%-adx-period_{period}"] = ta.ADX(dataframe, timeperiod=period)
        ema = ta.EMA(dataframe, timeperiod=period)
        dataframe[f"%-dist_ema-period_{period}"] = (dataframe["close"] - ema) / dataframe["close"]
        # MEAN-REVERSION — z-score + Bollinger %B (inline; no qtpylib dependency).
        roll_close = dataframe["close"].rolling(window=period, min_periods=period)
        mean = roll_close.mean()
        std = roll_close.std(ddof=0)
        dataframe[f"%-zscore-period_{period}"] = (dataframe["close"] - mean) / std
        typical_price = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
        tp_roll = typical_price.rolling(window=period, min_periods=period)
        tp_mid = tp_roll.mean()
        tp_sd = tp_roll.std(ddof=0)
        upper = tp_mid + 2.0 * tp_sd
        lower = tp_mid - 2.0 * tp_sd
        dataframe[f"%-bb_pctb-period_{period}"] = (dataframe["close"] - lower) / (upper - lower)
        # VOLATILITY — ATR% + realised vol (clustering, regime).
        dataframe[f"%-atr_pct-period_{period}"] = (
            ta.ATR(dataframe, timeperiod=period) / dataframe["close"]
        )
        dataframe[f"%-realized_vol-period_{period}"] = (
            dataframe["close"].pct_change().rolling(window=period, min_periods=period).std(ddof=0)
        )
        # VOLUME / MICROSTRUCTURE — participation + order-flow proxies.
        vol_roll = dataframe["volume"].rolling(window=period, min_periods=period)
        dataframe[f"%-vol_z-period_{period}"] = (
            dataframe["volume"] - vol_roll.mean()
        ) / vol_roll.std(ddof=0)
        dataframe[f"%-mfi-period_{period}"] = ta.MFI(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs: object
    ) -> DataFrame:
        """Period-independent basic features."""
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-obv"] = ta.OBV(dataframe)
        # VWAP distance over a fixed 96-candle window (intraday participation).
        typical_price = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
        tpv = (typical_price * dataframe["volume"]).rolling(window=96, min_periods=96).sum()
        vol_sum = dataframe["volume"].rolling(window=96, min_periods=96).sum()
        vwap = tpv / vol_sum
        dataframe["%-vwap_dist"] = (dataframe["close"] - vwap) / dataframe["close"]
        dataframe["%-raw_volume"] = dataframe["volume"]
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs: object
    ) -> DataFrame:
        """Calendar features (24/7 crypto seasonality)."""
        dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        dataframe["%-hour_of_day"] = (dataframe["date"].dt.hour + 1) / 24
        return dataframe

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs: object
    ) -> DataFrame:
        """Volatility-scaled triple-barrier classification target (research §B).

        For each bar ``i`` set an upper barrier at ``close * (1 + tb_profit_mult
        * vol)`` and a lower barrier at ``close * (1 - tb_stop_mult * vol)``,
        where ``vol`` is the rolling ``tb_vol_window`` std of returns. Walk
        forward up to ``label_period_candles`` (the vertical barrier): the first
        barrier touched decides the label — upper → "up", lower or neither →
        "down" (binary meta-label: was this a profitable long setup?). FreqAI
        strips the trailing ``label_period_candles`` rows (their forward look
        peeks past the window) so this is a legitimate label, not a leak.
        """
        n = self.label_period_candles
        vol = (
            dataframe["close"]
            .pct_change()
            .rolling(window=self.tb_vol_window, min_periods=self.tb_vol_window)
            .std(ddof=0)
        )
        close = dataframe["close"].to_numpy()
        high = dataframe["high"].to_numpy()
        low = dataframe["low"].to_numpy()
        up_unit = (self.tb_profit_mult * vol).to_numpy()
        dn_unit = (self.tb_stop_mult * vol).to_numpy()

        labels = np.full(len(dataframe), np.nan)
        for i in range(len(dataframe) - n):
            if not np.isfinite(up_unit[i]) or up_unit[i] == 0.0:
                continue
            up_px = close[i] * (1.0 + up_unit[i])
            dn_px = close[i] * (1.0 - dn_unit[i])
            hit = 0
            for j in range(1, n + 1):
                if high[i + j] >= up_px:
                    hit = 1
                    break
                if low[i + j] <= dn_px:
                    hit = -1
                    break
            labels[i] = hit

        dataframe["&-trade"] = np.where(labels == 1, "up", "down")
        dataframe.loc[~np.isfinite(labels), "&-trade"] = np.nan
        return dataframe

    # ─── Indicators (consumed by entry/exit + custom callbacks) ────────────

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Compute ATR / ADX / EMA200 used by the gate + callbacks, then FreqAI."""
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        # FreqAI populates do_predict, the predicted-label column &-trade, and one
        # probability column PER CLASS VALUE — "up" and "down" (NOT a
        # &-trade_<class>_proba name). See the module docstring NB.
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    # ─── Entries ───────────────────────────────────────────────────────────

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Long entry: model confidence + trend regime + EMA200 trend filter."""
        dataframe.loc[
            (
                (dataframe["do_predict"] == 1)
                & (dataframe["up"] > self.entry_proba)
                & (dataframe["adx"] > self.adx_min)
                & (dataframe["close"] > dataframe["ema200"])
                & (dataframe["volume"] > 0)  # exchange downtime guard
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "tb_ml_long")
        return dataframe

    # ─── Exits ─────────────────────────────────────────────────────────────

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Long exit: model flips bearish (P(up) < 0.5). ATR target/stop = callbacks."""
        dataframe.loc[
            ((dataframe["do_predict"] == 1) & (dataframe["up"] < 0.5)),
            "exit_long",
        ] = 1
        return dataframe

    # ─── ATR target / stop callbacks (research §D) ─────────────────────────

    def custom_stoploss(
        self,
        pair: str,
        trade: object,
        current_time: object,
        current_rate: float,
        current_profit: float,
        **kwargs: object,
    ) -> float | None:
        """ATR-scaled trailing stop: stop distance = atr_stop_mult * ATR / rate."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return None
        atr = dataframe["atr"].iloc[-1]
        if not np.isfinite(atr) or current_rate == 0:
            return None
        stop_distance = self.atr_stop_mult * atr / current_rate
        return -abs(float(stop_distance))

    def custom_exit(
        self,
        pair: str,
        trade: object,
        current_time: object,
        current_rate: float,
        current_profit: float,
        **kwargs: object,
    ) -> str | None:
        """ATR profit-take: exit when profit ≥ atr_target_mult * ATR / open_rate."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return None
        atr = dataframe["atr"].iloc[-1]
        if not np.isfinite(atr) or trade.open_rate == 0:  # type: ignore[attr-defined]
            return None
        target = self.atr_target_mult * atr / trade.open_rate  # type: ignore[attr-defined]
        if current_profit >= target:
            return "atr_target"
        return None
