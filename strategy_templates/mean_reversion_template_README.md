# mean_reversion_template — README

> **Hypothesis under test.** When the price of a deep, liquid crypto-spot
> pair stretches to the lower Bollinger Band while RSI is oversold,
> short-horizon mean reversion (price returns toward the BB middle) is
> more likely than continuation. The strategy buys those stretches and
> exits when price has reverted or RSI has rotated back up.

This is the v1 baseline template (BRD §8.1). Pure technical analysis, no
FreqAI, no ML. It exists to be **beaten** — the FreqAI classifier and
regressor templates (added in Stage 5) only justify their cost over this
baseline if they outperform it on the same walk-forward folds.

The critic agent (BRD §5.3) should attack the hypothesis directly using
this README as ammo. See "Why this can fail" below.

---

## Market belief encoded

1. **Spot crypto majors mean-revert on intraday timeframes.** 5m–1h on
   BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT is dominated by market-maker
   activity, retail panic, and stop-cascades — all of which produce
   temporary dislocations that revert.
2. **The lower Bollinger Band is a noisy-but-cheap "cheap" signal.** It's
   well-defined, robust to outliers (via rolling std), and survives the
   common look-ahead pitfalls when computed candle-close.
3. **RSI < 30 is the canonical oversold filter.** It rejects the common
   failure mode of "BB blow-out during a downtrend" — in a strong trend,
   RSI sits below 30 for long stretches without reverting, and entries on
   BB alone get knifed.
4. **Exit on the BB midline (or RSI rotation) is the natural complement.**
   It books the mean reversion without trying to catch the upper-band
   continuation move, which is a different regime.

---

## When this strategy should make money

- **Ranging markets with normal volatility.** Sideways price action with
  occasional stretches to the bands. This is BRD §5.4's "mid-vol regime".
- **Pairs with two-sided order flow.** Deep books mean BB stretches snap
  back; thin books mean BB stretches lead to liquidation cascades.

## When this strategy will lose money

- **Strong unidirectional trends.** "Lower band in a downtrend" is the
  reason BB-only entries are dangerous. The RSI filter mitigates but does
  not eliminate this.
- **Volatility regime shifts.** When realised vol jumps, the BB widens
  *after* the move, so entries during the initial spike are taken near
  the band's old (narrow) width and stopped out.
- **Pairs that decouple from BTC.** Mean reversion on alt/USDT works when
  the alt is correlated to BTC. When SOL or BNB diverges sharply, the
  reversion can be hours or days away.

---

## Slot table

These are the only parameters the generator can fill. Each appears as a
`# SLOT: <name> (type, range)` comment in
[mean_reversion_template.py](mean_reversion_template.py) and is enforced by
`MeanReversionParams` in
[mean_reversion_template_schema.py](mean_reversion_template_schema.py).

| Slot | Type | Range | Default | What it controls |
|---|---|---|---|---|
| `bb_period` | int | 10–50 | 20 | Bollinger Bands lookback window |
| `bb_std` | float | 1.5–3.0 | 2.0 | Standard-deviation multiplier for BB envelope |
| `rsi_period` | int | 7–30 | 14 | RSI lookback |
| `rsi_buy_threshold` | int | 10–40 | 30 | RSI must be below this for entry |
| `rsi_exit_threshold` | int | 50–80 | 55 | RSI above this triggers exit |
| `stoploss` | float | -0.10 to -0.02 | -0.05 | Hard per-trade stoploss |

The default literals are themselves inside the schema's ranges; the template
is a runnable strategy even without LLM rendering, which is what lets the
Stage 3 smoke test invoke it directly.

---

## Structural shell — never edited by the LLM

Per BRD §8 rule 1, the following are hand-written and frozen. The LLM does
not touch them; the AST validator (Stage 5) rejects any rendered output
that mutates them.

- Class name (`MeanReversionTemplate`)
- `timeframe = "5m"`
- `process_only_new_candles = True`
- `can_short = False` (BRD §1 — spot-only, long-only)
- `INTERFACE_VERSION = 3`
- `startup_candle_count = 200`
- `minimal_roi` (long ROI floor; strategy logic owns exits)
- `populate_indicators` body shape (RSI + BB columns)
- `populate_entry_trend` body shape (close ≤ lower BB ∧ RSI < threshold)
- `populate_exit_trend` body shape (close ≥ middle BB ∨ RSI > threshold)

The LLM may only change the *literal RHS values* on the six SLOT lines.

---

## Validation expectations

The Stage 3 integration test (`tests/integration/test_freqtrade_subprocess.py`)
runs this template on 1 week of cached BTC/USDT 5m data and asserts
`trades > 0`. That's a sanity floor, not a quality bar.

Real validation happens in Stage 4 (`validation` subgraph): anchored 6-fold
walk-forward, trade-level bootstrap, regime split, fee stress (BRD §5.4 +
§10). The gate thresholds the rendered strategy must clear live in
`orchestrator/gates/thresholds.py` (BRD §10).

---

## Why this can fail (critic ammo)

The critic agent (BRD §5.3, Opus 4.7) should look for at least these
specific weaknesses in any LLM-tuned variant of this strategy:

1. **Look-ahead on band width.** Confirm that `bb_*` columns use only
   data up to the current candle. The qtpylib helper used here is safe;
   any switch to `pandas.rolling().std()` without `min_periods` set is
   suspect.
2. **RSI on the wrong source.** `ta.RSI(dataframe, ...)` reads `close` by
   default — fine here. Any change to `open` or `hl2` is a red flag.
3. **Stoploss vs typical BB excursion.** If the LLM picks
   `stoploss=-0.02` with `bb_std=3.0`, a normal BB stretch is bigger than
   the stop and the strategy gets whipsawed.
4. **Entry on the same candle that triggers exit.** The default
   `process_only_new_candles=True` prevents this; any flip to `False`
   without justification is a red flag.
5. **Slot drift.** Pydantic schema and SLOT comments must agree. If the
   critic sees a SLOT in the rendered file without a matching schema
   field (or vice versa), the rendered template is invalid.
