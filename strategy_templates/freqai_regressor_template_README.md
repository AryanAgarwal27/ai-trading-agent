# freqai_regressor_template — README

> **Hypothesis under test.** A supervised regressor (LightGBM under
> FreqAI) trained on technical features can estimate the *magnitude* of
> the next `label_period_candles` return well enough that "enter when
> predicted_return > k * ATR%" produces a positive-expectancy long-only
> spot strategy after fees.

This is the second FreqAI template (BRD §8.1). It's the regression
counterpart to [freqai_classifier_template](freqai_classifier_template.py):
the classifier asks *which direction*, the regressor asks *how far*. The
two encode different priors about market predictability; backtests on
the same folds decide which (if either) earns its keep.

The critic agent (BRD §5.3, Opus 4.7) should attack the hypothesis using
this README as ammo. See "Why this can fail" below.

---

## Market belief encoded

1. **Forward return magnitude is partially predictable.** Even when
   direction is hard to call (which the classifier tries to exploit),
   the *size* of the move may carry information — e.g., compressed
   Bollinger bands often precede larger-than-average moves regardless
   of direction. A regressor that learns "I expect a ~0.4% up-move on
   average from this setup" can produce a useful entry signal that a
   classifier's discrete buckets erase.
2. **ATR-scaled thresholds keep the signal meaningful across regimes.**
   A 0.5% predicted return is a strong signal in a calm regime (ATR%
   ≈ 0.1%) and noise in a vol-of-vol regime (ATR% ≈ 1%). Comparing the
   prediction to `k * ATR%` instead of a fixed return floor adapts the
   filter automatically.
3. **A trend filter (EMA fast > EMA slow) gates the entry.** Same
   instinct as the classifier — even a strong positive prediction in a
   downtrend is "buying knives".
4. **Exit when the prediction turns negative OR trend inverts.** Two
   independent exit triggers — the model's own sign change, and the
   structural trend filter — keep the position from grinding to zero
   when the regime shifts.

---

## When this strategy should make money

- **Volatility-compression regimes that resolve to the upside.**
  Compressed BB → low ATR → low entry threshold → easy to clear with a
  moderately positive prediction. When such regimes break up (which the
  trend filter requires), the strategy enters near the breakout and
  rides it.
- **Trending pairs with low fee burn relative to predicted-return
  scale.** BTC and ETH on Binance with maker-only routing (0.02% fee)
  can profit from per-trade predicted returns in the 0.3–0.8% range.

## When this strategy will lose money

- **High-vol chop.** ATR is high → entry threshold is high → few
  entries → those that fire are on the largest predictions, which on
  noisy data are often outliers / mis-estimates.
- **Calibration drift.** The regressor's expected-return estimate may
  systematically diverge from realized return as the model ages between
  retrains (BRD §7.3 `live_retrain_hours=24`). The paper monitor must
  log realized-vs-predicted residuals; a drifting bias is a pause
  trigger.
- **Trend filter false negatives.** A genuine trend reversal at the
  start of a new uptrend often has `ema_fast < ema_slow` for the first
  few hours; the strategy misses the early ramp. Accepted trade-off —
  the alternative (entering before the trend filter confirms) takes
  more entries on false reversals.

---

## Slot table

These are the only parameters the generator can fill. Each appears as a
`# SLOT: <name> (type, range)` comment in
[freqai_regressor_template.py](freqai_regressor_template.py) and is
enforced by `FreqaiRegressorParams` in
[freqai_regressor_template_schema.py](freqai_regressor_template_schema.py).

| Slot | Type | Range | Default | What it controls |
|---|---|---|---|---|
| `ema_fast` | int | 5–50 | 12 | Fast EMA window (entry trend filter) |
| `ema_slow` | int | 20–200 | 50 | Slow EMA window (entry trend filter) |
| `atr_period` | int | 7–30 | 14 | ATR lookback for the entry threshold scale |
| `label_period_candles` | int | 4–24 | 12 | Forward window for the regression label |
| `k_atr_multiplier` | float | 0.5–3.0 | 1.5 | Entry threshold: predicted_return > k * (ATR/close) |
| `stoploss` | float | -0.10 to -0.02 | -0.05 | Hard per-trade stoploss |

The default literals are themselves inside the schema's ranges.

**Critic must check (cross-field):** `ema_fast < ema_slow`. Pydantic
field-level constraints can't span fields.

---

## Structural shell — never edited by the LLM

Per BRD §8 rule 1, the following are hand-written and frozen:

- Class name (`FreqaiRegressorTemplate`)
- `timeframe = "5m"`, `process_only_new_candles = True`, `can_short = False`
- `INTERFACE_VERSION = 3`, `startup_candle_count = 200`
- `minimal_roi` (ROI exits disabled; strategy logic owns exits)
- `freqai_config` (BRD §7.3 pins: train/backtest period, retrain cadence,
  expiration, DI threshold, SVM outlier rejection)
- `feature_engineering_expand_all` body (RSI, ROC, ATR, BB-width features)
- `feature_engineering_expand_basic` body (pct_change, raw_volume)
- `feature_engineering_standard` body (day_of_week, hour_of_day)
- `set_freqai_targets` body (fractional forward return as the regression label)
- `populate_indicators` body shape (EMA / ATR + FreqAI handoff)
- `populate_entry_trend` body shape (do_predict + prediction beats ATR + trend)
- `populate_exit_trend` body shape (prediction sign flip OR trend inversion)

The LLM may only change the *literal RHS values* on the six SLOT lines.

---

## Validation expectations

Same gauntlet as the classifier (BRD §5.4, §10). FreqAI-specific checks:

- `freqtrade lookahead-analysis` must pass (BRD §8 rule 5).
- Realized-vs-predicted residual distribution on the OOS fold should be
  approximately mean-zero. A persistent positive bias means the regressor
  systematically overestimates returns and entries will lose money even
  when the predicted-return signal "looks good".
- ATR-threshold rejection rate: % of bars where `predicted_return >
  k * ATR%` is false. If rejection rate > 99% across the OOS fold, `k`
  is too high for this regime and the strategy is effectively flat.

---

## Why this can fail (critic ammo)

The critic agent should look for at least these specific weaknesses:

1. **Inverted EMA ordering.** `ema_fast >= ema_slow` reverses the trend
   filter; the schema doesn't catch it.
2. **`k_atr_multiplier` too low.** Below 0.5 means even a marginally
   positive prediction triggers an entry — the strategy degenerates
   into "trade on any positive noise above zero".
3. **`k_atr_multiplier` too high.** Above 2.5 in a calm regime
   (ATR% ≈ 0.1%) requires a predicted return > 0.25% on every entry —
   the model rarely emits these, the strategy is effectively flat, and
   the operator sees "no entries" with no clear reason.
4. **`label_period_candles` × fees > realized predictable move.** A
   4-candle (20-minute) forward return on 5m has expected magnitude in
   the same band as round-trip fees on Binance taker (0.2% round-trip).
   The regressor "learns" but the strategy can't profit from what it
   learned.
5. **Calibration drift between retrains.** The regressor predicts well
   immediately after retrain and degrades over the 24-hour cycle. Paper
   monitor must log per-hour residual mean/std; if mean drifts far
   from zero before the next retrain, the strategy is overdue.
6. **Trend filter eats the early entry.** On a clean regime change,
   `ema_fast < ema_slow` for the first hours of the new uptrend; the
   strategy misses the early ramp. This is by design — the alternative
   eats more false reversals — but the critic should confirm the trade
   was made consciously and not accidentally.
