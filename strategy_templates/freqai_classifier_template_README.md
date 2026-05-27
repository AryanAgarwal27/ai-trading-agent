# freqai_classifier_template — README

> **Hypothesis under test.** A supervised classifier (LightGBM under
> FreqAI) trained on technical features can discriminate short-horizon
> forward returns into `{up, flat, down}` with enough precision that
> "trade only on 'up' above `min_class_prob`" beats the pure-TA
> mean-reversion baseline after fees on the same walk-forward folds.

This is the first FreqAI template (BRD §8.1). It must justify its
operating cost (training, inference, retrain cadence) by outperforming
[mean_reversion_template](mean_reversion_template.py) on the same data
under the same gate thresholds. If it doesn't, the regressor template is
the next thing to try; if neither does, "no FreqAI in v1" is the right
call and the operator should hear that from the critic.

The critic agent (BRD §5.3, Opus 4.7) should attack the hypothesis using
this README as ammo. See "Why this can fail" below.

---

## Market belief encoded

1. **Forward 5m–1h direction is partially predictable from recent TA.**
   RSI / MFI / ROC / Bollinger-Band width over multiple lookback windows
   carry information about whether the next `label_period_candles` close
   is likely above, below, or near the current close. The "near" bucket
   is intentional — it absorbs the dominant random-walk component so the
   classifier doesn't waste capacity on it.
2. **The 3-class formulation is a regularizer.** A binary up/down
   classifier overfits the noise band around zero return. The 'flat'
   bucket is a sink for "I don't know" and keeps `min_class_prob` on the
   'up' class meaningful as a real confidence.
3. **A trend filter (EMA fast > EMA slow) gates the entry.** The
   classifier may emit high-conviction 'up' in a downtrend; entering
   then is "buying knives". The trend filter is the same instinct
   experienced operators apply by eye.
4. **RSI-oversold layer on top of the classifier.** Two independent
   signals agreeing (ML says 'up', RSI says oversold) is a sharper
   filter than either alone. The cost is fewer entries — accepted.

---

## When this strategy should make money

- **Trending markets with periodic shallow pullbacks.** EMA filter says
  "we're up", RSI dip says "right now is a pullback", classifier says
  "the next leg is up". This is the cleanest setup.
- **Pairs with persistent autocorrelation in returns.** Some 5m series
  show real, decay-fast momentum that an L0/L1-regularized GBM can
  exploit. BTC and ETH are the most likely; SOL and BNB less so.

## When this strategy will lose money

- **Regime breaks during a paper window.** A model trained on a calm
  month can't predict a vol-of-vol spike. BRD §5.4's regime split
  catches this in backtest; the live `regime_check` reviewer catches it
  in real time.
- **Label leakage via lookahead in features.** Any feature that
  inadvertently uses future data trains a fantastic classifier that
  collapses live. `freqtrade lookahead-analysis` is the AST validator's
  partner — both run before paper.
- **DI threshold too lax.** With `DI_threshold=0.9` (BRD §7.3 default)
  the model rejects far-OOD inputs. Lowering it widens prediction surface
  at the cost of trading on points the model never really learned from.
- **Probability column missing.** If a Freqtrade version bump changes
  the per-class probability column name from `&-action_up_proba` to
  something else, entries silently never fire. The paper monitor must
  alert on "zero entries with `do_predict=1` for ≥ N candles".

---

## Slot table

These are the only parameters the generator can fill. Each appears as a
`# SLOT: <name> (type, range)` comment in
[freqai_classifier_template.py](freqai_classifier_template.py) and is
enforced by `FreqaiClassifierParams` in
[freqai_classifier_template_schema.py](freqai_classifier_template_schema.py).

| Slot | Type | Range | Default | What it controls |
|---|---|---|---|---|
| `rsi_period` | int | 7–30 | 14 | RSI lookback (entry filter only) |
| `rsi_buy_threshold` | int | 10–40 | 25 | RSI must be below this to enter long |
| `ema_fast` | int | 5–50 | 12 | Fast EMA window (entry trend filter) |
| `ema_slow` | int | 20–200 | 50 | Slow EMA window (entry trend filter) |
| `label_period_candles` | int | 4–24 | 12 | Forward window for classifier label |
| `label_threshold_pct` | float | 0.1–2.0 | 0.6 | |return| threshold (%) separating up/down from flat |
| `min_class_prob` | float | 0.55–0.85 | 0.7 | Required probability on 'up' class |
| `stoploss` | float | -0.10 to -0.02 | -0.05 | Hard per-trade stoploss |

The default literals are themselves inside the schema's ranges.

**Critic must check (cross-field):** `ema_fast < ema_slow`. Pydantic
field-level constraints can't span fields; if the LLM picks
`ema_fast=40, ema_slow=30`, the entry trend filter inverts and the
strategy trades against its own thesis.

---

## Structural shell — never edited by the LLM

Per BRD §8 rule 1, the following are hand-written and frozen:

- Class name (`FreqaiClassifierTemplate`)
- `timeframe = "5m"`, `process_only_new_candles = True`, `can_short = False`
- `INTERFACE_VERSION = 3`, `startup_candle_count = 200`
- `minimal_roi` (ROI exits disabled; strategy logic owns exits)
- `freqai_config` (BRD §7.3 pins: train/backtest period, retrain cadence,
  expiration, DI threshold, SVM outlier rejection)
- `feature_engineering_expand_all` body (RSI, MFI, ROC, BB-width features)
- `feature_engineering_expand_basic` body (pct_change, raw_volume, raw_price)
- `feature_engineering_standard` body (day_of_week, hour_of_day)
- `set_freqai_targets` body (3-class label from forward return)
- `populate_indicators` body shape (EMA / RSI + FreqAI handoff)
- `populate_entry_trend` body shape (do_predict + 'up' + min_prob + trend + RSI)
- `populate_exit_trend` body shape (class flip OR trend inversion)

The LLM may only change the *literal RHS values* on the eight SLOT lines.

---

## Validation expectations

Stage 5 unit tests (`tests/unit/test_template_filling.py`) verify
SLOT/schema alignment and AST cleanliness. Stage 4 integration tests
(`validation` subgraph) run the rendered strategy through anchored 6-fold
walk-forward, trade-level bootstrap, regime split, and fee stress (BRD
§5.4 + §10). The gate thresholds the rendered strategy must clear live in
[orchestrator/gates/thresholds.py](../orchestrator/gates/thresholds.py).

For FreqAI specifically, the additional pre-paper checks are:

- `freqtrade lookahead-analysis` must pass (BRD §8 rule 5).
- The model's training set must contain ≥ 1 occurrence of each label
  class in the train window — a degenerate label distribution (e.g.
  100% 'flat' on a low-vol month) means the classifier learned nothing.
- DI threshold rejection rate on the OOS fold must be reported; if
  > 50% of OOS bars are filtered as OOD, the model is mismatched to
  current regime and the strategy should not advance to paper.

---

## Why this can fail (critic ammo)

The critic agent should look for at least these specific weaknesses:

1. **Inverted EMA ordering.** `ema_fast >= ema_slow` reverses the trend
   filter; the schema doesn't catch it (no cross-field constraint).
2. **Label window too short relative to fees.** A 4-candle (20-minute)
   forward return on 5m has expected magnitude in the same band as
   round-trip fees — the classifier "learns" but the strategy can't
   profit from what it learned.
3. **`min_class_prob` too low.** Below 0.6 on a 3-class problem is
   barely above the no-information rate (0.33); entries are essentially
   random above noise.
4. **`min_class_prob` too high.** Above 0.8 on out-of-sample data is
   almost never emitted by a well-calibrated classifier; the strategy
   takes zero trades and degenerates into "do nothing".
5. **Feature leakage in `set_freqai_targets`.** The body uses
   `dataframe["close"].shift(-self.label_period_candles)` — that's a
   *forward* shift, which is correct *as a label* (FreqAI strips the
   tail before training). Any switch to using shifted values as features
   instead is a leak.
6. **Probability-column rename.** The entry logic reads
   `&-action_up_proba`; if a Freqtrade version bump renames this,
   entries silently never fire. Paper monitor must alert on this.
7. **DI threshold mismatch with regime.** A model trained on calm
   conditions sees high-vol bars as OOD and rejects them; the operator
   sees "no entries" and assumes the strategy is broken. The regime
   reviewer should flag "OOD rejection rate spiked" as a pause trigger.
