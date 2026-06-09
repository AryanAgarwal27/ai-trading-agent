"""Unit tests for orchestrator.tools.freqai_config (BRD §7.3, §21.3).

Pure AST + dict building — no Docker, no import of the strategy files (they
import freqtrade/talib which only resolve in the container).
"""

from __future__ import annotations

from pathlib import Path

from orchestrator.tools.freqai_config import (
    build_freqai_config,
    extract_class_int,
    extract_class_number,
    extract_freqai_pins,
    extract_model_training_parameters,
    freqai_model_for,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEMPLATES = REPO_ROOT / "strategy_templates"
REGRESSOR = TEMPLATES / "freqai_regressor_template.py"
MEAN_REVERSION = TEMPLATES / "mean_reversion_template.py"
TRIPLE_BARRIER = TEMPLATES / "freqai_triple_barrier_template.py"


def test_extract_freqai_pins_on_freqai_template() -> None:
    pins = extract_freqai_pins(REGRESSOR)
    assert pins is not None
    assert pins["train_period_days"] == 30
    assert pins["backtest_period_days"] == 7
    assert pins["feature_parameters"]["DI_threshold"] == 0.9
    assert pins["feature_parameters"]["use_SVM_to_remove_outliers"] is True


def test_extract_freqai_pins_on_non_freqai_returns_none() -> None:
    """mean_reversion is pure-TA → no freqai_config attribute → None (the
    detection signal that the backtest path stays non-FreqAI)."""
    assert extract_freqai_pins(MEAN_REVERSION) is None


def test_extract_class_int_label_period() -> None:
    # Template slot default (the generator fills a concrete value per strategy).
    assert extract_class_int(REGRESSOR, "label_period_candles") == 12
    assert extract_class_int(REGRESSOR, "nonexistent_attr") is None


def test_freqai_model_for_keyed_off_class_name() -> None:
    assert freqai_model_for("FreqaiClassifierTemplate") == "LightGBMClassifier"
    assert freqai_model_for("FreqaiRegressorTemplate") == "LightGBMRegressor"
    # Unknown FreqAI class → regressor default.
    assert freqai_model_for("SomeFutureFreqaiThing") == "LightGBMRegressor"


def test_freqai_model_for_triple_barrier_is_classifier() -> None:
    """Requirement: the triple-barrier template resolves to the CLASSIFIER.

    The class name carries the "Classifier" substring on purpose so the runner
    selects ``--freqaimodel LightGBMClassifier``, not the regressor default.
    """
    assert freqai_model_for("FreqaiTripleBarrierClassifier") == "LightGBMClassifier"


# ─── Stage 12: triple-barrier template — slot extraction + builder ──────


def test_extract_class_number_reads_float_and_rejects_bool() -> None:
    # Float SLOT (the regressor's k_atr_multiplier default is 1.5).
    assert extract_class_number(REGRESSOR, "k_atr_multiplier") == 1.5
    # Int SLOT still works through the generalised reader.
    assert extract_class_number(REGRESSOR, "label_period_candles") == 12
    # Bool class attribute (can_short = False) is NOT a number → None.
    assert extract_class_number(REGRESSOR, "can_short") is None
    assert extract_class_number(REGRESSOR, "nonexistent") is None


def test_extract_class_number_reads_triple_barrier_float_slots() -> None:
    assert extract_class_number(TRIPLE_BARRIER, "DI_threshold") == 0.9
    assert extract_class_number(TRIPLE_BARRIER, "weight_factor") == 0.5
    assert extract_class_number(TRIPLE_BARRIER, "learning_rate") == 0.02
    assert extract_class_number(TRIPLE_BARRIER, "label_period_candles") == 36


def test_extract_model_training_parameters_from_triple_barrier() -> None:
    """The 9 LightGBM SLOTs are read off the class into model_training_parameters,
    plus bagging_freq=1 so bagging_fraction actually takes effect."""
    mtp = extract_model_training_parameters(TRIPLE_BARRIER)
    assert mtp["n_estimators"] == 200
    assert mtp["learning_rate"] == 0.02
    assert mtp["num_leaves"] == 16
    assert mtp["max_depth"] == 4
    assert mtp["min_child_samples"] == 80
    assert mtp["lambda_l1"] == 1.0
    assert mtp["lambda_l2"] == 1.0
    assert mtp["feature_fraction"] == 0.7
    assert mtp["bagging_fraction"] == 0.7
    assert mtp["bagging_freq"] == 1


def test_extract_model_training_parameters_empty_for_non_tb_template() -> None:
    """The regressor exposes no model SLOTs → empty dict (LightGBM defaults)."""
    assert extract_model_training_parameters(REGRESSOR) == {}


def test_build_freqai_config_triple_barrier_carries_research_block() -> None:
    """End-to-end: the runtime config carries the full research block — multi-TF
    feature scope from the pins + model_training_parameters/DI/weight from the
    SLOT class attributes (single source of truth, no drift)."""
    pins = extract_freqai_pins(TRIPLE_BARRIER)
    assert pins is not None
    cfg = build_freqai_config(
        pins,
        timeframe="5m",
        label_period_candles=36,
        identifier="FreqaiTripleBarrierClassifier",
        strategy_path=TRIPLE_BARRIER,
    )
    fp = cfg["feature_parameters"]
    # Pinned feature scope honoured.
    assert fp["include_timeframes"] == ["5m", "1h"]
    assert fp["include_corr_pairlist"] == ["BTC/USDT", "ETH/USDT"]
    assert fp["indicator_periods_candles"] == [14, 48]
    assert fp["use_SVM_to_remove_outliers"] is True
    # DI_threshold + weight_factor come from the SLOT class attributes.
    assert fp["DI_threshold"] == 0.9
    assert fp["weight_factor"] == 0.5
    # Model params assembled from the LightGBM SLOTs.
    mtp = cfg["model_training_parameters"]
    assert mtp["n_estimators"] == 200
    assert mtp["max_depth"] == 4
    assert mtp["bagging_freq"] == 1
    # Time-series-forecasting hygiene preserved.
    assert cfg["data_split_parameters"]["shuffle"] is False


def test_build_freqai_config_triple_barrier_is_1h_capable() -> None:
    """At a 1h base the sub-base 5m informative TF is dropped (FreqAI rejects an
    informative timeframe faster than the base) — the same template runs at 1h."""
    pins = extract_freqai_pins(TRIPLE_BARRIER)
    assert pins is not None
    cfg = build_freqai_config(
        pins,
        timeframe="1h",
        label_period_candles=36,
        identifier="FreqaiTripleBarrierClassifier",
        strategy_path=TRIPLE_BARRIER,
    )
    assert cfg["feature_parameters"]["include_timeframes"] == ["1h"]


def test_build_freqai_config_regressor_unchanged_with_strategy_path() -> None:
    """Regression guard: passing strategy_path for the regressor (no model SLOTs,
    DI_threshold lives in its pins) leaves the runtime config identical to the
    pre-Stage-12 behaviour."""
    pins = extract_freqai_pins(REGRESSOR)
    assert pins is not None
    cfg = build_freqai_config(
        pins,
        timeframe="5m",
        label_period_candles=12,
        identifier="FreqaiRegressorTemplate",
        strategy_path=REGRESSOR,
    )
    fp = cfg["feature_parameters"]
    assert fp["include_timeframes"] == ["5m"]
    assert fp["include_corr_pairlist"] == []
    assert fp["DI_threshold"] == 0.9  # from pins, not a class attr
    assert cfg["model_training_parameters"] == {}


def test_build_freqai_config_carries_pins_and_adds_runtime_fields() -> None:
    pins = extract_freqai_pins(REGRESSOR)
    assert pins is not None
    cfg = build_freqai_config(
        pins,
        timeframe="5m",
        label_period_candles=12,
        identifier="FreqaiRegressorTemplate",
    )
    # Runtime-required fields Freqtrade validates.
    assert cfg["enabled"] is True
    assert cfg["identifier"] == "FreqaiRegressorTemplate"
    fp = cfg["feature_parameters"]
    assert fp["include_timeframes"] == ["5m"]
    assert fp["include_corr_pairlist"] == []
    assert fp["label_period_candles"] == 12
    assert len(fp["indicator_periods_candles"]) >= 1
    # BRD §7.3 pins carried verbatim.
    assert cfg["train_period_days"] == 30
    assert cfg["backtest_period_days"] == 7
    assert fp["DI_threshold"] == 0.9
    assert fp["use_SVM_to_remove_outliers"] is True
    assert "data_split_parameters" in cfg
    assert "model_training_parameters" in cfg
