"""Pydantic schema co-located with ``freqai_triple_barrier_template.py`` (BRD §8 rule 3).

The generator (BRD §5.3) and the manual-inject endpoint (BRD §21.2) validate
params against this schema, so neither the LLM nor the operator can emit a value
outside these ranges. Free-form parameter generation is impossible by
construction.

The fields here MUST mirror the ``# SLOT:`` markers in the template
byte-for-byte (same name, same type, same closed interval) — enforced by
``tests/unit/test_template_filling.py::test_slot_names_match_schema_fields``. The
template's literal defaults are also legal per these constraints, so the
un-rendered template is itself a runnable strategy.

Ranges are verbatim from the research design (§D2 tunable-parameter table): the
triple-barrier + meta-gate logic params, plus the LightGBM regularisation knobs
the research stresses for overfitting control on ~8,640-candle (30-day 5m)
training windows.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FreqaiTripleBarrierParams(BaseModel):
    """Validated parameter set for the FreqAI triple-barrier classifier template.

    Two groups (research §D2):

    *Strategy logic* — ``entry_proba`` (meta-gate P(up) threshold), ``adx_min``
    (trend-regime floor), ``tb_profit_mult`` / ``tb_stop_mult`` / ``tb_vol_window``
    (vol-scaled triple-barrier geometry in ``set_freqai_targets``),
    ``atr_target_mult`` / ``atr_stop_mult`` (ATR exit/stop callbacks),
    ``label_period_candles`` (vertical-barrier + FreqAI label horizon — research
    §B pushes this to hours-not-minutes, 24–72 candles).

    *Model / FreqAI regularisation* — ``n_estimators`` … ``bagging_fraction`` are
    the LightGBM knobs; the research keeps trees shallow (``max_depth`` 3–6,
    ``num_leaves`` ≪ 2^max_depth), with strong L1/L2 and high ``min_child_samples``
    to fight leaf-wise overfitting on a small window. ``DI_threshold`` activates
    FreqAI's Dissimilarity-Index outlier refusal; ``weight_factor`` recency-weights
    training data. These eleven are SLOTS (class attributes) that
    ``build_freqai_config`` reads off the rendered template and assembles into the
    runtime ``model_training_parameters`` / ``feature_parameters`` — the slot is
    the single source of truth, never a literal duplicated in a config dict.
    """

    # `extra='forbid'` and `frozen=True` guard against silent slot drift between
    # template and schema: a stray field in the output would raise, not persist.
    model_config = ConfigDict(extra="forbid", frozen=True)

    # ─── Strategy-logic params ─────────────────────────────────────────────
    entry_proba: float = Field(
        ge=0.50, le=0.80, description="Minimum classifier P(up) to fire an entry."
    )
    adx_min: int = Field(ge=15, le=35, description="ADX floor — only trade a trending regime.")
    tb_profit_mult: float = Field(
        ge=1.0, le=4.0, description="Triple-barrier profit-take width, in units of realised vol."
    )
    tb_stop_mult: float = Field(
        ge=1.0, le=4.0, description="Triple-barrier stop width, in units of realised vol."
    )
    tb_vol_window: int = Field(
        ge=20, le=100, description="Rolling window (candles) for the barrier-scaling realised vol."
    )
    atr_target_mult: float = Field(
        ge=1.0, le=4.0, description="ATR profit-take multiple (custom_exit)."
    )
    atr_stop_mult: float = Field(ge=1.0, le=3.0, description="ATR stop multiple (custom_stoploss).")
    label_period_candles: int = Field(
        ge=24,
        le=72,
        description="Vertical-barrier + FreqAI label horizon (candles of `timeframe`).",
    )

    # ─── Model / FreqAI regularisation params ──────────────────────────────
    n_estimators: int = Field(ge=100, le=400, description="LightGBM boosting rounds.")
    learning_rate: float = Field(ge=0.005, le=0.05, description="LightGBM learning rate.")
    num_leaves: int = Field(ge=8, le=31, description="LightGBM leaves (≪ 2^max_depth — overfit).")
    max_depth: int = Field(ge=3, le=6, description="LightGBM tree depth (shallow = anti-overfit).")
    min_child_samples: int = Field(
        ge=40, le=150, description="LightGBM min_data_in_leaf (key overfit guard)."
    )
    lambda_l1: float = Field(ge=0.0, le=5.0, description="LightGBM L1 regularisation.")
    lambda_l2: float = Field(ge=0.0, le=5.0, description="LightGBM L2 regularisation.")
    feature_fraction: float = Field(
        ge=0.5, le=0.9, description="LightGBM per-tree feature subsample."
    )
    bagging_fraction: float = Field(
        ge=0.5, le=0.9, description="LightGBM per-iteration row subsample."
    )
    DI_threshold: float = Field(
        ge=0.0, le=2.0, description="FreqAI Dissimilarity-Index outlier-refusal threshold."
    )
    weight_factor: float = Field(
        ge=0.0, le=1.0, description="FreqAI recency-weighting of training data."
    )
