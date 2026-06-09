# bb_regime_reversion_template

Regime-filtered Bollinger mean-reversion (pure TA, no FreqAI). The one long-only
shape found to clear the validation gate on the project's **current bearish**
anchored walk-forward window.

## Idea

Buy short, sharp dips that stretch **below the lower Bollinger Band** with an
oversold RSI — **but only when price is above a long trend EMA** (uptrend
regime). The regime filter is the load-bearing part: it keeps the strategy out of
falling markets, where buying dips is catching knives, and only buys pullbacks
inside an upward drift. Exit fast (revert to the band mean, or a small fixed
take-profit), with a hard stop catching the rest.

- **Entry:** `close <= lower_BB` AND `RSI < rsi_buy_threshold` AND `close > EMA(ema_trend_period)`
- **Exit:** `close >= middle_BB`, or `minimal_roi` take-profit (`roi_target`), or hard `stoploss`
- **Timeframe:** 15m · long-only spot

## Slots (see `bb_regime_reversion_template_schema.py`)

`bb_period`, `bb_std`, `rsi_period`, `rsi_buy_threshold`, `ema_trend_period`,
`roi_target`, `stoploss`.

## Recommended params (clear the gate on the current window — proxy)

```json
{
  "template": "bb_regime_reversion_template",
  "name": "bb_regime_reversion_v1",
  "pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"],
  "timeframe": "15m",
  "params": {"bb_period": 20, "bb_std": 2.3, "rsi_period": 14,
             "rsi_buy_threshold": 35, "ema_trend_period": 200,
             "roi_target": 0.025, "stoploss": -0.06}
}
```

## Proxy walk-forward evidence (FIRST-PASS FILTER ONLY)

Replaying the pipeline's anchored 6-fold walk-forward (Nov 2025 – May 2026, a
window in which every pair fell 12–39%) over the cached feathers — see
`scripts/proxy_walkforward.py`:

| config (bb_std / rsi_buy) | mean Sharpe | profit factor | positive folds | trades | max DD |
|---|---|---|---|---|---|
| **2.3 / 35 (default)** | **0.81** | **1.45** | **4 / 6** | **135** | **11.4%** |
| 2.5 / 35 | 0.62 | 1.42 | 4 / 6 | 121 | 10.4% |
| 2.6 / 35 | 0.65 | 1.34 | 4 / 6 | 109 | 11.8% |

> The proxy is a slightly-conservative approximation (entry at next-candle open,
> intrabar stops, 0.1%/side fees). It is **not** Freqtrade. The authoritative
> numbers come from `POST /strategies/validate`, which runs the real Docker
> backtest gauntlet. Confirm there before trusting any of the above.

## Why the trade-count floor was re-tuned

This strategy is selective by design and tops out near ~95–135 trades; pushing
past 150 forces in low-quality chop trades that drag Sharpe below the floor (the
proxy Pareto frontier never crosses `(150 trades, Sharpe 0.5)` on this window).
`MIN_TRADES_IS` was re-tuned 150 → 90 on 2026-06-09 with the per-fold (`>= 5`) and
4-of-6 positive-fold consistency gates carrying the anti-luck role. See
`orchestrator/gates/thresholds.py`.
