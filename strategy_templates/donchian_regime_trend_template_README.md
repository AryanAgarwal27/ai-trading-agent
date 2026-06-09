# donchian_regime_trend_template

Regime-filtered Donchian breakout (pure TA, no FreqAI). The **complementary**
trend strategy to the mean-reversion template — built to harvest sustained
directional moves.

## Idea

In a confirmed uptrend (fast EMA above slow EMA), a break above the recent
N-candle high signals trend continuation. Ride it with a **trailing stop** so
winners run and losers are cut early; bail when the trend regime flips.

- **Entry:** `close > prior_N_high` AND `EMA(ema_fast) > EMA(ema_slow)`
- **Exit:** `EMA(ema_fast) < EMA(ema_slow)` (regime flip), trailing stop, or hard `stoploss`
- **Timeframe:** 1h · long-only spot · ROI disabled (let winners run)

## Slots (see `donchian_regime_trend_template_schema.py`)

`donchian_period`, `ema_fast`, `ema_slow`, `trailing_stop_positive`, `stoploss`.

## ⚠️ Regime note — does NOT pass on the current window

Trend-following needs trends. The current anchored walk-forward window
(Nov 2025 – May 2026) is bearish/choppy — every pair fell 12–39%, with a −25%
crash in fold 3. Long breakouts get chopped up here; the proxy shows the best
config at **Sharpe ≈ −2.2, drawdown 27%+**, failing the gate. No parameter set
fixes a regime mismatch.

This template is shipped so the portfolio has a trend strategy **ready to promote
when the anchored window contains an up-trending regime** (re-run
`POST /strategies/validate` after the OHLCV cache advances into a bull phase).
It is the right tool for a different market, deliberately kept in the roster for
diversification across regimes.

## Recommended params (starting point for a trending window)

```json
{
  "template": "donchian_regime_trend_template",
  "name": "donchian_regime_trend_v1",
  "pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"],
  "timeframe": "1h",
  "params": {"donchian_period": 20, "ema_fast": 20, "ema_slow": 50,
             "trailing_stop_positive": 0.05, "stoploss": -0.10}
}
```
