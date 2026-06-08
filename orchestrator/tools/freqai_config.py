"""Shared FreqAI runtime-config builder (BRD §7.3, §21.3 design-check).

A FreqAI strategy calls ``self.freqai.start(...)`` in ``populate_indicators``,
which Freqtrade refuses to run unless the runtime ``config.json`` carries a
``freqai`` block with ``enabled: true`` plus the feature/model parameters. The
strategy templates ship a hand-written ``freqai_config`` CLASS ATTRIBUTE
carrying only the BRD §7.3 PINS (train/backtest windows, retrain/expiration,
outlier rejection) — NOT the full runtime block Freqtrade validates.

This module is the single seam that turns those pins into a complete ``freqai``
config. It is used by the backtest runner now (so manual-injected / supervisor
FreqAI strategies actually backtest instead of erroring "freqAI is not enabled");
the paper/live spawn path should reuse the SAME builder when it grows FreqAI
support, so the backtest, paper, and live FreqAI configs can never drift —
mirroring the Stage 12 shared-seam approach (BRD §21.1).

The model class (``LightGBMRegressor`` / ``LightGBMClassifier``) is NOT part of
the ``freqai`` config block — Freqtrade selects it via the ``--freqaimodel`` CLI
arg (see :func:`freqai_model_for`). Both ship in the ``stable_freqai`` image.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

# The two shipped FreqAI templates (BRD §8.1) map to these models. Default to the
# regressor for any other FreqAI strategy (e.g. a future template) — a wrong model
# would surface a clear Freqtrade error now that stderr is captured (commit 1).
_CLASSIFIER_MODEL = "LightGBMClassifier"
_REGRESSOR_MODEL = "LightGBMRegressor"

# FreqAI iterates these periods in ``feature_engineering_expand_all`` (the
# strategy's per-period RSI/ROC/ATR/bb_width features). Kept inside the strategy's
# ``startup_candle_count`` (200) so no partial-window look-ahead.
_DEFAULT_INDICATOR_PERIODS: tuple[int, ...] = (10, 20)


def _class_def(strategy_path: Path) -> ast.ClassDef | None:
    """Return the first top-level ``ClassDef`` in the strategy file, or None."""
    tree = ast.parse(strategy_path.read_text(encoding="utf-8"), filename=str(strategy_path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            return node
    return None


def _assigned_value(stmt: ast.stmt, name: str) -> ast.expr | None:
    """Return the RHS expr of ``<name> = ...`` / ``<name>: T = ...``, else None."""
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        if stmt.target.id == name and stmt.value is not None:
            return stmt.value
    elif isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return stmt.value
    return None


def extract_freqai_pins(strategy_path: Path) -> dict[str, Any] | None:
    """Return the strategy's ``freqai_config`` class-attribute dict, or None.

    None means the strategy is NOT FreqAI (no ``freqai_config`` attribute) — the
    "class exposes freqai_config" detection criterion (BRD §21.3). Uses
    ``ast.literal_eval`` so the strategy file is never imported (it imports
    ``freqtrade`` / ``talib``, which only resolve inside the container).
    """
    cls = _class_def(strategy_path)
    if cls is None:
        return None
    for stmt in cls.body:
        value = _assigned_value(stmt, "freqai_config")
        if value is not None:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, dict) else None
    return None


def extract_class_int(strategy_path: Path, attr: str) -> int | None:
    """Return an int class attribute (e.g. ``label_period_candles``), or None."""
    cls = _class_def(strategy_path)
    if cls is None:
        return None
    for stmt in cls.body:
        value = _assigned_value(stmt, attr)
        if value is not None:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, int) and not isinstance(parsed, bool) else None
    return None


def freqai_model_for(strategy_class: str) -> str:
    """Select the ``--freqaimodel`` keyed off the template's class name (BRD §8.1).

    ``FreqaiClassifierTemplate`` → ``LightGBMClassifier``; the regressor (and any
    other FreqAI strategy) → ``LightGBMRegressor``. The class name is derived from
    the template, so this is "keyed off the template" without threading the
    template name down into the runner.
    """
    return _CLASSIFIER_MODEL if "classifier" in strategy_class.lower() else _REGRESSOR_MODEL


def build_freqai_config(
    pins: dict[str, Any],
    *,
    timeframe: str,
    label_period_candles: int,
    identifier: str,
    indicator_periods_candles: tuple[int, ...] = _DEFAULT_INDICATOR_PERIODS,
) -> dict[str, Any]:
    """Build the runtime ``freqai`` config block from the strategy's §7.3 pins.

    Carries the pins verbatim (train/backtest windows, retrain/expiration, purge,
    and the pins' ``feature_parameters`` — DI_threshold + outlier rejection) and
    adds the runtime fields Freqtrade REQUIRES that the pins omit:
    ``enabled``, ``identifier``, ``feature_parameters.include_timeframes`` (the
    base timeframe), ``include_corr_pairlist`` ([]), ``label_period_candles``
    (which MUST match the strategy's slot), ``indicator_periods_candles`` (the
    periods FreqAI iterates in ``feature_engineering_expand_all``),
    ``data_split_parameters`` and ``model_training_parameters``.
    """
    pin_features = dict(pins.get("feature_parameters") or {})
    feature_parameters: dict[str, Any] = {
        "include_timeframes": [timeframe],
        "include_corr_pairlist": [],
        "label_period_candles": label_period_candles,
        "include_shifted_candles": 2,
        "indicator_periods_candles": list(indicator_periods_candles),
        # Pins' feature params (DI_threshold, use_SVM_to_remove_outliers) last so
        # the BRD §7.3 values win on any overlap.
        **pin_features,
    }
    return {
        "enabled": True,
        "identifier": identifier,
        "train_period_days": pins.get("train_period_days", 30),
        "backtest_period_days": pins.get("backtest_period_days", 7),
        "live_retrain_hours": pins.get("live_retrain_hours", 24),
        "expiration_hours": pins.get("expiration_hours", 72),
        "purge_old_models": pins.get("purge_old_models", 2),
        "feature_parameters": feature_parameters,
        "data_split_parameters": {"test_size": 0.33, "shuffle": False},
        "model_training_parameters": {},
    }
