"""Pydantic schema co-located with ``mean_reversion_template.py`` (BRD §8 rule 3).

The generator (BRD §5.3) calls ``ChatAnthropic(...).with_structured_output(
MeanReversionParams)`` so the LLM can only emit values inside these ranges.
Free-form parameter generation is impossible by construction.

The fields here MUST mirror the ``# SLOT:`` markers in the template byte-for-byte
(same name, same type, same closed interval). The template's literal default
values are also legal per these constraints — the rendered template stays valid
even before slot substitution, which is what lets the Stage 3 / Stage 5 smoke
tests backtest the un-rendered baseline.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MeanReversionParams(BaseModel):
    """Validated parameter set for the mean-reversion template.

    Ranges are tuned to keep entries plausibly profitable on 5m crypto:
      - ``bb_period`` 10–50: shorter is noisier, longer lags too much.
      - ``bb_std`` 1.5–3.0: tighter bands fire too often; wider bands almost
        never fire on 5m.
      - ``rsi_period`` 7–30: 7 is the standard fast setting, 30 is the long end.
      - ``rsi_buy_threshold`` 10–40: ≤30 is canonical oversold, the LLM may
        push more aggressive entries below 20 on high-vol regimes.
      - ``rsi_exit_threshold`` 50–80: 50 is "regression to mean", 70+ waits
        for momentum to flip.
      - ``stoploss`` -0.10 to -0.02: BRD §10 caps max drawdown at 20%; a
        per-trade stop tighter than -2% would whipsaw out of normal noise.
    """

    # `extra='forbid'` and `frozen=True` guard against silent slot drift between
    # template and schema: a stray field in the LLM output would raise instead
    # of being persisted.
    model_config = ConfigDict(extra="forbid", frozen=True)

    bb_period: int = Field(ge=10, le=50, description="Bollinger Bands lookback window.")
    bb_std: float = Field(
        ge=1.5,
        le=3.0,
        description="Bollinger Bands standard-deviation multiplier.",
    )
    rsi_period: int = Field(ge=7, le=30, description="RSI lookback.")
    rsi_buy_threshold: int = Field(
        ge=10, le=40, description="RSI must be below this to enter long."
    )
    rsi_exit_threshold: int = Field(ge=50, le=80, description="RSI above this triggers exit.")
    stoploss: float = Field(
        ge=-0.10,
        le=-0.02,
        description="Hard per-trade stoploss as a negative fraction.",
    )
