"""Pydantic schema co-located with ``bb_regime_reversion_template.py`` (BRD §8 rule 3).

The generator calls ``ChatAnthropic(...).with_structured_output(BbRegimeReversionParams)``
so the LLM can only emit values inside these ranges. The manual-injection endpoint
(POST /strategies/validate) validates operator-supplied params against the same model.

Fields MUST mirror the ``# SLOT:`` markers in the template byte-for-byte (same name,
same type, same closed interval). The template's literal defaults are legal per these
constraints, so the un-rendered baseline still backtests.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BbRegimeReversionParams(BaseModel):
    """Validated parameter set for the regime-filtered Bollinger reversion template.

    Ranges (tuned on the cached 5m/15m crypto data, anchored 6-fold walk-forward):
      - ``bb_period`` 10-50: shorter is noisier, longer lags too much.
      - ``bb_std`` 1.5-3.0: the edge concentrates around 2.3-2.6 (deep, selective
        dips); below ~2.2 the extra entries are low-quality chop.
      - ``rsi_period`` 7-30: 14 is canonical.
      - ``rsi_buy_threshold`` 10-45: ~35 is the robust sweet spot; above ~38 the
        strategy over-trades and quality degrades in down regimes.
      - ``ema_trend_period`` 50-400: the regime filter. 200 is the sweet spot — long
        enough to define trend, short enough to re-enable trading after a recovery.
      - ``roi_target`` 0.005-0.05: fixed take-profit. ~0.025 (2.5%) is robust; the
        result is insensitive across 0.02-0.03 because the mean-reversion exit usually
        fires first.
      - ``stoploss`` -0.10 to -0.02: a hard floor; in practice rarely hit because the
        regime filter + mean-reversion exit close most trades first.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    bb_period: int = Field(ge=10, le=50, description="Bollinger Bands lookback window.")
    bb_std: float = Field(
        ge=1.5, le=3.0, description="Bollinger Bands standard-deviation multiplier."
    )
    rsi_period: int = Field(ge=7, le=30, description="RSI lookback.")
    rsi_buy_threshold: int = Field(
        ge=10, le=45, description="RSI must be below this to enter long."
    )
    ema_trend_period: int = Field(
        ge=50, le=400, description="Uptrend-regime EMA; only buy dips above it."
    )
    roi_target: float = Field(
        ge=0.005, le=0.05, description="Fixed take-profit fraction (minimal_roi)."
    )
    stoploss: float = Field(
        ge=-0.10, le=-0.02, description="Hard per-trade stoploss as a negative fraction."
    )
