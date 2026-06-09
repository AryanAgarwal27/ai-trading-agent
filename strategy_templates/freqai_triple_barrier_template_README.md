# freqai_triple_barrier_template — README

> **Hypothesis under test.** Replacing the forward-return *regression*
> target with a volatility-scaled *triple-barrier classification* target
> (López de Prado), feeding a compact feature set grounded in documented
> crypto effects, and gating entries on a meta-labelling rule
> (`P(up) > entry_proba` in a trending regime above EMA200) produces a
> long-only spot FreqAI strategy that clears fees on BTC/ETH/SOL/BNB —
> where the raw-return regressor scored Sharpe −24 to −105.

This is the third FreqAI template (BRD §8.1), added in Stage 12 from the
attached research brief. It is the **answer to a diagnosed failure**, not
a fresh idea: the regressor's raw forward-return target at 5m is
noise-dominated, so a gradient-boosted model minimises MSE by predicting
near the unconditional mean (~0), the entry rule fires near-randomly, and
each random round trip pays ~0.15–0.2% in Binance fees — hundreds of
fee-bleeding trades compound into deeply negative Sharpe.

The critic agent (BRD §5.3, Opus 4.7) should attack the hypothesis using
this README as ammo. See "Why this can fail" below — including the
research's own honest verdict that this design likely lands
**~breakeven**, not at a glossy Sharpe ≥ 1.5.

---

## Market belief encoded

1. **A path-dependent, vol-normalised target is learnable where a raw
   forward return is not.** The triple-barrier label asks "did a
   vol-scaled profit-take get hit before a vol-scaled stop, within
   `label_period_candles`?" — a balanced, regime-normalised
   classification problem that matches how the trade is actually managed
   (stops/targets), rather than a near-zero-mean regression on noise.
2. **Classification beats regression in low-SNR finance.** A calibrated
   `P(up)` gives a natural, tunable confidence gate (`entry_proba`);
   the model only needs to rank setups, not estimate magnitudes.
3. **Quality features over quantity.** ~50–80 columns from families with
   real empirical grounding — time-series momentum (Liu & Tsyvinski),
   BTC→alt lead-lag (via `include_corr_pairlist`), volatility clustering,
   and volume/VWAP participation — across 5m **and** 1h
   (`include_timeframes`) to raise signal-to-noise.
4. **Trade less, trade only high-conviction setups.** At 5m the per-trade
   edge is tiny and fees are fixed, so trade-frequency control is as
   important as prediction quality: the ADX regime filter gates out chop,
   `entry_proba` filters to high-conviction longs, and the hours-not-
   minutes `label_period_candles` (24–72) lets moves clear fees.
5. **ATR-scaled exits.** `custom_exit` takes profit at
   `atr_target_mult * ATR`; `custom_stoploss` trails at
   `atr_stop_mult * ATR` — both adapt to the current volatility regime.

---

## When this strategy should make money

- **Trending regimes the ADX + EMA200 filters admit.** Long-only edge in
  crypto is regime-dependent; gating out chop is the single most reliable
  improvement (research §Recommendations stage 3).
- **1h base timeframe.** The research is explicit that 5m is structurally
  the hardest place to clear fees; the same template at `timeframe="1h"`
  (config-driven — `build_freqai_config` adapts `include_timeframes`)
  has higher SNR and fewer fees per unit signal.

## When this strategy will lose money

- **5m majors, most of the time.** The research's honest verdict: Sharpe
  ≥ 1.5 here is improbable; expect ~breakeven-to-modestly-positive after
  fees at best. The realistic benchmark is "beats buy-and-hold of the 4
  coins" (DEFERRED.md D-20), not an abstract Sharpe number.
- **Featureless / noise-dominated windows.** If the features genuinely
  lack signal on these 4 coins at this timeframe, no plumbing manufactures
  alpha — the meta-gate can only improve an existing signal's precision.
- **Overfit on the ~8,640-candle window.** FreqAI multiplies features
  combinatorially; on 30 days of 5m data that is an overfitting trap.

---

## Slot table (research §D2)

The only parameters the generator / manual-inject can fill. Each appears
as a `# SLOT: <name> (type, range)` comment in
[freqai_triple_barrier_template.py](freqai_triple_barrier_template.py) and
is enforced by `FreqaiTripleBarrierParams` in
[freqai_triple_barrier_template_schema.py](freqai_triple_barrier_template_schema.py).

### Strategy logic (read by the strategy methods)

| Slot | Type | Range | Default | Controls |
|---|---|---|---|---|
| `entry_proba` | float | 0.50–0.80 | 0.60 | Min `P(up)` to enter (meta-gate) |
| `adx_min` | int | 15–35 | 20 | ADX floor — trade only a trending regime |
| `tb_profit_mult` | float | 1.0–4.0 | 2.0 | Triple-barrier profit-take (vol units) |
| `tb_stop_mult` | float | 1.0–4.0 | 1.5 | Triple-barrier stop (vol units) |
| `tb_vol_window` | int | 20–100 | 48 | Rolling window for barrier-scaling vol |
| `atr_target_mult` | float | 1.0–4.0 | 2.0 | ATR profit-take (custom_exit) |
| `atr_stop_mult` | float | 1.0–3.0 | 1.5 | ATR stop (custom_stoploss) |
| `label_period_candles` | int | 24–72 | 36 | Vertical-barrier + FreqAI label horizon |

### Model / FreqAI regularisation

These eleven are SLOTS (class attributes) but the *strategy* never reads
them: `build_freqai_config`
([orchestrator/tools/freqai_config.py](../orchestrator/tools/freqai_config.py))
reads them off the rendered class and assembles them into the runtime
`model_training_parameters` / `feature_parameters`. One source of truth —
the slot — never a literal duplicated in the `freqai_config` dict.

| Slot | Type | Range | Default | Controls |
|---|---|---|---|---|
| `n_estimators` | int | 100–400 | 200 | LightGBM boosting rounds |
| `learning_rate` | float | 0.005–0.05 | 0.02 | LightGBM learning rate |
| `num_leaves` | int | 8–31 | 16 | Leaves (≪ 2^max_depth — overfit guard) |
| `max_depth` | int | 3–6 | 4 | Tree depth (shallow = anti-overfit) |
| `min_child_samples` | int | 40–150 | 80 | min_data_in_leaf (key overfit guard) |
| `lambda_l1` | float | 0.0–5.0 | 1.0 | L1 regularisation |
| `lambda_l2` | float | 0.0–5.0 | 1.0 | L2 regularisation |
| `feature_fraction` | float | 0.5–0.9 | 0.7 | Per-tree feature subsample |
| `bagging_fraction` | float | 0.5–0.9 | 0.7 | Per-iteration row subsample |
| `DI_threshold` | float | 0.0–2.0 | 0.9 | FreqAI Dissimilarity-Index outlier refusal |
| `weight_factor` | float | 0.0–1.0 | 0.5 | FreqAI recency-weighting of training data |

The default literals are themselves inside the schema's ranges.

**Critic must check (cross-field):** `tb_stop_mult` materially smaller
than `tb_profit_mult` skews the labels toward "up" (easy upper barrier)
and inflates apparent precision; `num_leaves` near `2^max_depth`
re-opens the overfit door the shallow-tree design closes.

---

## Structural shell — never edited by the LLM

Per BRD §8 rule 1, the following are hand-written and frozen:

- Class name (`FreqaiTripleBarrierClassifier` — the "Classifier" substring
  is load-bearing: `freqai_model_for` resolves `--freqaimodel
  LightGBMClassifier` off it).
- `timeframe = "5m"` (config-overridable for 1h), `process_only_new_candles
  = True`, `can_short = False`, `use_custom_stoploss = True`.
- `INTERFACE_VERSION = 3`, `startup_candle_count = 300` (EMA200 + vol/label
  windows), `stoploss = -0.05` (hard backstop; ATR stop is primary).
- `minimal_roi` (ROI exits disabled; ATR + model logic own exits).
- `freqai_config` (BRD §7.3 pins + feature scope: 5m/1h timeframes,
  BTC/ETH corr pairs, indicator periods, SVM outlier rejection).
- `feature_engineering_*` bodies (the ~50–80-column feature set).
- `set_freqai_targets` body (the vol-scaled triple-barrier label).
- `populate_indicators` / `populate_entry_trend` / `populate_exit_trend`
  body shapes, and `custom_stoploss` / `custom_exit`.

The LLM may only change the *literal RHS values* on the 19 SLOT lines.

---

## Validation expectations

Same gauntlet as the other FreqAI templates (BRD §5.4, §10, as re-tuned
2026-06-08 — SPEC §6). FreqAI-specific checks:

- `freqtrade lookahead-analysis` must pass (BRD §8 rule 5). The
  triple-barrier loop and rolling features are classic lookahead traps —
  only `set_freqai_targets` may reference forward bars (the barrier walk),
  and FreqAI strips the trailing `label_period_candles` rows so it is a
  legitimate label, not a leak.
- The classifier must actually train and emit the `up` probability column.
  The Stage 12 freqtrade-marked integration test runs one real
  train→backtest fold — a FreqAI-version rename of the probability column
  surfaces there as a `KeyError` in `populate_entry_trend`, not silently as
  "zero entries". (It already earned its keep: the first draft guessed
  `&-trade_up_proba` and the test caught it — the real column is `up`.)
- Treat any Sharpe ≥ 1.5 with deep suspicion (Deflated Sharpe Ratio;
  42 days OOS). The gate is positive-in-≥4/6-folds + PF > 1.2 (SPEC §6).

---

## Why this can fail (critic ammo)

1. **The label leaks if a `shift(-` ever appears outside
   `set_freqai_targets`.** The barrier walk legitimately reads forward
   bars; any forward reference in a `feature_engineering_*` body is a
   silent leak. Audit: grep `shift(-` / forward indexing in the feature
   bodies of any rendered file.
2. **Asymmetric barriers manufacture precision.** `tb_profit_mult` ≪
   `tb_stop_mult` makes the upper barrier trivially easy to hit → labels
   skew "up" → the classifier looks accurate but is predicting a tautology.
3. **`entry_proba` too low** degenerates into "trade on any positive
   noise"; **too high** → no entries, operator sees a silent flat line.
4. **`num_leaves` near `2^max_depth`** (LightGBM docs warn this overfits)
   undoes the shallow-tree regularisation the design depends on.
5. **Fee drag still wins at 5m.** Even with the better target, the
   per-trade edge may not clear ~0.15–0.2% round-trip. The honest move
   (research §E, recommendation 4) is to test 1h before over-optimising 5m
   — forcing a 1.5 on 5m yields an overfit backtest that loses live.
6. **Single-class training windows.** A fold where no upper barrier is hit
   yields an all-"down" target; the classifier has one class and predicts
   a constant. The per-fold consistency gate (SPEC §6 `MIN_POSITIVE_FOLDS`)
   surfaces this as a failing fold rather than a hidden degenerate model.
7. **Probability-column name drift.** For a string target `&-trade`,
   freqtrade 2026.4 names the per-class probability columns after the class
   VALUES (`up`, `down`) — so the entry reads `df["up"]`, not a
   `&-trade_up_proba` form (which a wrong-guess first draft tripped on). A
   version bump that renames the column makes the entry `KeyError` rather
   than silently never fire. The integration test is the canary — keep it
   green.
