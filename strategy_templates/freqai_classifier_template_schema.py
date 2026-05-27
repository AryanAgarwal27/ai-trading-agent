"""Pydantic schema co-located with ``freqai_classifier_template.py`` (BRD §8 rule 3).

The generator (BRD §5.3) calls
``ChatAnthropic(...).with_structured_output(FreqaiClassifierParams)`` so the
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


class FreqaiClassifierParams(BaseModel):
    """Validated parameter set for the FreqAI 3-class predictor template.

    Ranges are tuned to keep the classifier learnable on 5m crypto without
    overfitting:
      - ``rsi_period`` 7–30: standard band; the same channel as mean-reversion.
      - ``rsi_buy_threshold`` 10–40: oversold filter on top of the classifier
        signal — see ``freqai_classifier_template_README.md`` for why we layer
        these.
      - ``ema_fast`` 5–50 / ``ema_slow`` 20–200: trend filter; the schema does
        NOT enforce ``ema_fast < ema_slow`` because Pydantic field-level
        constraints can't span fields. The critic agent (BRD §5.3) is
        responsible for catching inverted EMA configs.
      - ``label_period_candles`` 4–24: forward window for the training label;
        too short → noise dominates, too long → fewer samples per regime.
      - ``label_threshold_pct`` 0.1–2.0: |return| threshold (percent) that
        separates 'up'/'down' from 'flat'. Tighter → more 'flat' labels, fewer
        but higher-precision entries.
      - ``min_class_prob`` 0.55–0.85: a healthy 3-class classifier rarely emits
        > 0.85 conviction on out-of-sample data; > 0.85 is usually a red flag
        of overfit or leak.
      - ``stoploss`` -0.10 to -0.02: BRD §10 caps max drawdown at 20%; a stop
        tighter than -2% would whipsaw out of normal 5m noise.
    """

    # `extra='forbid'` and `frozen=True` guard against silent slot drift between
    # template and schema: a stray field in the LLM output would raise instead
    # of being persisted.
    model_config = ConfigDict(extra="forbid", frozen=True)

    rsi_period: int = Field(ge=7, le=30, description="RSI lookback.")
    rsi_buy_threshold: int = Field(
        ge=10,
        le=40,
        description="RSI must be below this to enter long (oversold filter).",
    )
    ema_fast: int = Field(ge=5, le=50, description="Fast EMA window (entry trend filter).")
    ema_slow: int = Field(
        ge=20,
        le=200,
        description="Slow EMA window. Should be > ema_fast (critic enforces).",
    )
    label_period_candles: int = Field(
        ge=4,
        le=24,
        description="Forward window (candles of `timeframe`) for the training label.",
    )
    label_threshold_pct: float = Field(
        ge=0.1,
        le=2.0,
        description="|forward_return| > this (percent) → 'up'/'down'; else 'flat'.",
    )
    min_class_prob: float = Field(
        ge=0.55,
        le=0.85,
        description="Required probability on the 'up' class for entry.",
    )
    stoploss: float = Field(
        ge=-0.10,
        le=-0.02,
        description="Hard per-trade stoploss as a negative fraction.",
    )
