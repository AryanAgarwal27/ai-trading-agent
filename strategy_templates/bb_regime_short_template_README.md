# bb_regime_short_template

Regime-filtered Bollinger **SHORT** (pure TA, no FreqAI). The short-side mirror of
`bb_regime_reversion_template`, added in **Stage 13 Phase 1 (BRD §22.1)** so the
validation gauntlet can test whether **shorting** has edge on the project's
**bearish** anchored walk-forward window (Nov 2025–May 2026; every pair fell
12–39%) — a move a long-only spot system structurally cannot capture.

## Idea

Short sharp rallies that stretch **above the upper Bollinger Band** with an
overbought RSI — **but only when price is below a long trend EMA** (downtrend
regime). The regime filter is the load-bearing part: it keeps the strategy out of
rising markets, where shorting strength is fighting the trend, and only fades
exhaustion rallies inside a downward drift. Cover fast (revert to the band mean, or
a small fixed take-profit), with a hard stop catching the rest.

- **Entry (short):** `close >= upper_BB` AND `RSI > rsi_sell_threshold` AND `close < EMA(ema_trend_period)`
- **Exit (cover):** `close <= middle_BB`, or `minimal_roi` take-profit (`roi_target`), or hard `stoploss`
- **Timeframe:** 15m · `can_short = True`

## Execution scope — **Phase 1: BACKTEST/VALIDATION ONLY** (BRD §22.1)

- `can_short = True` requires a **futures** backtest config. The backtest runner
  detects this and renders `trading_mode="futures"` + `margin_mode="isolated"` for
  this strategy only (the long/spot path stays byte-identical). **Leverage is 1×**
  (the Freqtrade default — this template defines no `leverage()` callback).
- A short-capable strategy that passes the gauntlet is **blocked from promotion to
  the spot paper/live path** at `paper_spawn` (status
  `short_paper_deferred_to_phase2`). It can be **validated** but **cannot trade
  real money** until Stage 13 **Phase 2** ships the live futures/margin risk model
  (BRD §22.2). Nothing here touches a live account.
- Funding-rate cost is a real short cost and is accounted in the futures backtest
  P&L (requires the operator to download `funding_rate` + `mark` candles — see the
  P1-8 data-download step in BRD §22.3).

## Slots (see `bb_regime_short_template_schema.py`)

`bb_period`, `bb_std`, `rsi_period`, `rsi_sell_threshold`, `ema_trend_period`,
`roi_target`, `stoploss`.

## Starting params (to be validated — no proxy run performed)

```json
{
  "template": "bb_regime_short_template",
  "name": "bb_regime_short_v1",
  "pairs": ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT"],
  "timeframe": "15m",
  "params": {"bb_period": 20, "bb_std": 2.3, "rsi_period": 14,
             "rsi_sell_threshold": 70, "ema_trend_period": 200,
             "roi_target": 0.025, "stoploss": -0.06}
}
```

> **No proxy walk-forward numbers are recorded here.** Unlike the long
> `bb_regime_reversion_template`, this short template has **not** been triaged on
> the standalone proxy simulator — the short side needs futures + funding data the
> proxy does not model. The **authoritative** pass/fail + Sharpe/PF/drawdown
> figures come only from `POST /strategies/validate` (the real Docker futures
> backtest gauntlet, P1-9 in BRD §22.3) once the operator has downloaded the
> futures/funding/mark data (P1-8). Do not trust any performance claim for this
> template until that run completes.
>
> Note the **futures pair notation** (`BTC/USDT:USDT`) in the params block: a
> futures backtest whitelists the settled-perpetual symbol, not the spot
> `BTC/USDT`. The backtest config builder converts SPEC §1 Q2 spot pairs to this
> form for short-capable strategies.
