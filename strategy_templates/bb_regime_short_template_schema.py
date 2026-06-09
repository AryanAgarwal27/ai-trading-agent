"""Pydantic schema co-located with ``bb_regime_short_template.py`` (BRD §8 rule 3).

The generator calls ``ChatAnthropic(...).with_structured_output(BbRegimeShortParams)``
so the LLM can only emit values inside these ranges. The manual-injection endpoint
(POST /strategies/validate) — the Stage 13 Phase-1 entry point for short strategies
(BRD §22.1) — validates operator-supplied params against the same model.

Fields MUST mirror the ``# SLOT:`` markers in the template byte-for-byte (same name,
same type, same closed interval). The template's literal defaults are legal per these
constraints, so the un-rendered baseline still backtests.

This is the short-side mirror of ``bb_regime_reversion_template_schema.py``: the only
structural difference is the entry-RSI slot — ``rsi_sell_threshold`` (overbought, high
RSI) replaces the long template's ``rsi_buy_threshold`` (oversold, low RSI).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BbRegimeShortParams(BaseModel):
    """Validated parameter set for the regime-filtered Bollinger SHORT template.

    Ranges (mirror of the long regime-reversion template, short side):
      - ``bb_period`` 10-50: shorter is noisier, longer lags too much.
      - ``bb_std`` 1.5-3.0: the edge concentrates around 2.3-2.6 (deep, selective
        rallies); below ~2.2 the extra entries are low-quality chop.
      - ``rsi_period`` 7-30: 14 is canonical.
      - ``rsi_sell_threshold`` 55-90: RSI must be ABOVE this to short. ~70 is the
        robust overbought sweet spot; below ~62 the strategy over-trades and
        quality degrades.
      - ``ema_trend_period`` 50-400: the regime filter. 200 is the sweet spot — long
        enough to define the downtrend, short enough to re-enable after a leg down.
      - ``roi_target`` 0.005-0.05: fixed take-profit. ~0.025 (2.5%) is robust; the
        result is insensitive across 0.02-0.03 because the mean-reversion (cover)
        exit usually fires first.
      - ``stoploss`` -0.10 to -0.02: a hard floor; for a short the adverse move is
        price rising. In practice rarely hit because the regime filter + cover exit
        close most trades first.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    bb_period: int = Field(ge=10, le=50, description="Bollinger Bands lookback window.")
    bb_std: float = Field(
        ge=1.5, le=3.0, description="Bollinger Bands standard-deviation multiplier."
    )
    rsi_period: int = Field(ge=7, le=30, description="RSI lookback.")
    rsi_sell_threshold: int = Field(
        ge=55, le=90, description="RSI must be above this to enter short (overbought)."
    )
    ema_trend_period: int = Field(
        ge=50, le=400, description="Downtrend-regime EMA; only short rallies below it."
    )
    roi_target: float = Field(
        ge=0.005, le=0.05, description="Fixed take-profit fraction (minimal_roi)."
    )
    stoploss: float = Field(
        ge=-0.10, le=-0.02, description="Hard per-trade stoploss as a negative fraction."
    )
