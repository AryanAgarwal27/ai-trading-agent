"""Unit tests for orchestrator.tools.freqai_config (BRD §7.3, §21.3).

Pure AST + dict building — no Docker, no import of the strategy files (they
import freqtrade/talib which only resolve in the container).
"""

from __future__ import annotations

from pathlib import Path

from orchestrator.tools.freqai_config import (
    build_freqai_config,
    extract_class_int,
    extract_freqai_pins,
    freqai_model_for,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEMPLATES = REPO_ROOT / "strategy_templates"
REGRESSOR = TEMPLATES / "freqai_regressor_template.py"
MEAN_REVERSION = TEMPLATES / "mean_reversion_template.py"


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
