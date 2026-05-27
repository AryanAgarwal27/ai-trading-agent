"""Pydantic schema co-located with ``freqai_regressor_template.py`` (BRD §8 rule 3).

The generator (BRD §5.3) calls
``ChatAnthropic(...).with_structured_output(FreqaiRegressorParams)`` so the
LLM can only emit values inside these ranges. Free-form parameter generation
is impossible by construction.

The fields here MUST mirror the ``# SLOT:`` markers in the template
byte-for-byte (same name, same type, same closed interval). The template's
literal default values are also legal per these constraints — the rendered
template stays valid even before slot substitution, which is what lets the
Stage 5 template-filling smoke test render mid-range synthetic params and
re-validate.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FreqaiRegressorParams(BaseModel):
    """Validated parameter set for the FreqAI regression template.

    Ranges chosen to keep the regressor learnable on 5m crypto without
    overfitting:
      - ``ema_fast`` 5–50 / ``ema_slow`` 20–200: trend filter; schema does
        NOT enforce ``ema_fast < ema_slow`` — critic (BRD §5.3) catches.
      - ``atr_period`` 7–30: standard ATR band; same instinct as RSI period.
      - ``label_period_candles`` 4–24: forward window for the regression
        label. Too short → predicted return is dominated by microstructure
        noise and rarely clears the ATR threshold; too long → fewer
        independent samples per regime and the model degrades.
      - ``k_atr_multiplier`` 0.5–3.0: how many ATR%s the predicted return
        must beat to enter. Below 0.5 essentially trades on any positive
        prediction (noise); above 3.0 takes almost no trades.
      - ``stoploss`` -0.10 to -0.02: same band as the other templates.
    """

    # `extra='forbid'` and `frozen=True` guard against silent slot drift between
    # template and schema: a stray field in the LLM output would raise instead
    # of being persisted.
    model_config = ConfigDict(extra="forbid", frozen=True)

    ema_fast: int = Field(ge=5, le=50, description="Fast EMA window (entry trend filter).")
    ema_slow: int = Field(
        ge=20,
        le=200,
        description="Slow EMA window. Should be > ema_fast (critic enforces).",
    )
    atr_period: int = Field(ge=7, le=30, description="ATR lookback (entry threshold scale).")
    label_period_candles: int = Field(
        ge=4,
        le=24,
        description="Forward window (candles of `timeframe`) for the regression label.",
    )
    k_atr_multiplier: float = Field(
        ge=0.5,
        le=3.0,
        description="Entry threshold: predicted_return > k * (ATR / close).",
    )
    stoploss: float = Field(
        ge=-0.10,
        le=-0.02,
        description="Hard per-trade stoploss as a negative fraction.",
    )
