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


def extract_class_bool(strategy_path: Path, attr: str) -> bool | None:
    """Return a ``bool`` class attribute (e.g. ``can_short``), or None.

    Generic strategy-introspection reader — lives here alongside
    :func:`extract_class_int` / :func:`extract_class_number` (the module's other
    AST class-attribute readers) rather than in its own module. Used by the
    backtest runner / lookahead / paper_spawn to detect ``can_short = True``
    (Stage 13 Phase 1, BRD §22.1) the same way FreqAI detection reads pins off
    the rendered strategy — the strategy file is the single source of truth.

    ``ast.literal_eval`` only resolves a literal ``True`` / ``False``; a computed
    or referenced value returns None (treated as "not statically True" by
    :func:`strategy_can_short`, which fails closed to long-only).
    """
    cls = _class_def(strategy_path)
    if cls is None:
        return None
    for stmt in cls.body:
        value = _assigned_value(stmt, attr)
        if value is not None:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, bool) else None
    return None


def strategy_can_short(strategy_path: Path) -> bool:
    """True iff the strategy class statically sets ``can_short = True``.

    The single source of truth for "is this a short-capable strategy" (BRD
    §22.1). Fails CLOSED: anything other than a literal ``True`` — absent
    attribute, ``False``, or a non-literal expression — is long-only. Reading
    the (rendered) strategy file keeps this in lockstep with what Freqtrade
    actually runs, exactly like the FreqAI ``extract_freqai_pins`` detection.
    """
    return extract_class_bool(strategy_path, "can_short") is True


def extract_class_number(strategy_path: Path, attr: str) -> int | float | None:
    """Return an int OR float class attribute, or None.

    Generalises :func:`extract_class_int` to the float SLOTs the triple-barrier
    template exposes (``learning_rate``, ``DI_threshold``, …). ``bool`` is
    rejected (it subclasses ``int``) so a ``True``/``False`` attribute never
    leaks in as ``1``/``0``.
    """
    cls = _class_def(strategy_path)
    if cls is None:
        return None
    for stmt in cls.body:
        value = _assigned_value(stmt, attr)
        if value is not None:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, bool):
                return None
            if isinstance(parsed, int | float):
                return parsed
    return None


# LightGBM ``model_training_parameters`` that ship as SLOTs on a FreqAI template
# (research §D2 — the triple-barrier classifier). Read OFF the rendered class so
# the slot is the single source of truth; any FreqAI template exposing these
# class attributes gets them carried into the runtime config. The regressor /
# classifier do not expose them → empty dict → LightGBM defaults (unchanged).
_MODEL_PARAM_SLOTS: tuple[str, ...] = (
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "lambda_l1",
    "lambda_l2",
    "feature_fraction",
    "bagging_fraction",
)


def extract_model_training_parameters(strategy_path: Path) -> dict[str, Any]:
    """Assemble ``model_training_parameters`` from the template's SLOT class attrs.

    Returns only the model params present on the class (so a template without
    them yields ``{}`` and LightGBM uses its own defaults). When ANY are present
    it also sets ``bagging_freq=1`` — ``bagging_fraction`` is inert without a
    non-zero ``bagging_freq``, so the research's row-subsample only takes effect
    once bagging is enabled.
    """
    out: dict[str, Any] = {}
    for name in _MODEL_PARAM_SLOTS:
        value = extract_class_number(strategy_path, name)
        if value is not None:
            out[name] = value
    if out:
        out["bagging_freq"] = 1
    return out


def _timeframe_minutes(timeframe: str) -> int:
    """Convert a Freqtrade timeframe string (``"5m"`` / ``"1h"`` / ``"1d"``) to minutes."""
    unit = timeframe[-1].lower()
    value = int(timeframe[:-1])
    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 24 * 60
    raise ValueError(f"unsupported timeframe unit in {timeframe!r}")


def _resolve_include_timeframes(pin_value: Any, base_timeframe: str) -> list[str]:
    """Runtime ``include_timeframes``: base TF + the pin's strictly-higher TFs.

    FreqAI requires the base/strategy timeframe to be present and rejects an
    informative timeframe FASTER than the base. So a template pinning
    ``["5m", "1h"]`` runs as ``["5m", "1h"]`` at a 5m base but ``["1h"]`` at a 1h
    base (the sub-base 5m is dropped) — which is what makes the triple-barrier
    template 1h-capable from the same file (research §E). A missing/empty pin
    falls back to just the base timeframe (the regressor/classifier behaviour).
    """
    if not pin_value or not isinstance(pin_value, list | tuple):
        return [base_timeframe]
    base_min = _timeframe_minutes(base_timeframe)
    out = [base_timeframe]
    for tf in pin_value:
        if isinstance(tf, str) and tf not in out and _timeframe_minutes(tf) > base_min:
            out.append(tf)
    return out


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
    strategy_path: Path | None = None,
) -> dict[str, Any]:
    """Build the runtime ``freqai`` config block from the strategy's §7.3 pins.

    Carries the pins verbatim (train/backtest windows, retrain/expiration, purge,
    and the pins' ``feature_parameters`` — DI_threshold + outlier rejection) and
    adds the runtime fields Freqtrade REQUIRES that the pins omit:
    ``enabled``, ``identifier``, ``feature_parameters.include_timeframes``,
    ``include_corr_pairlist``, ``label_period_candles`` (which MUST match the
    strategy's slot), ``indicator_periods_candles``, ``data_split_parameters``
    and ``model_training_parameters``.

    ``include_timeframes`` / ``include_corr_pairlist`` are taken from the pins
    when present (a template may pin a multi-timeframe / corr-pair feature scope,
    e.g. the triple-barrier classifier's ``["5m", "1h"]`` + BTC/ETH); the
    timeframes are resolved against the base timeframe so the same template runs
    at 1h (see :func:`_resolve_include_timeframes`). A template that pins neither
    (the regressor / classifier) gets ``[timeframe]`` + ``[]`` — unchanged.

    ``strategy_path`` (optional): when given, the tunable SLOT class attributes
    are read OFF the rendered template and assembled into the runtime config —
    ``model_training_parameters`` from the LightGBM SLOTs and
    ``feature_parameters.{DI_threshold,weight_factor}`` overriding any pin. This
    keeps the SLOT the single source of truth for the tunable subset (no literal
    duplicated in the ``freqai_config`` dict). A template without those class
    attributes (the regressor, whose ``DI_threshold`` is a pin) is unaffected.
    """
    pin_features = dict(pins.get("feature_parameters") or {})
    # include_timeframes / include_corr_pairlist are resolved explicitly (the
    # base timeframe must be present, sub-base TFs dropped), so pop them out of
    # the generic pin merge below.
    include_timeframes = _resolve_include_timeframes(
        pin_features.pop("include_timeframes", None), timeframe
    )
    pin_corr = pin_features.pop("include_corr_pairlist", None)
    include_corr_pairlist = (
        [p for p in pin_corr if isinstance(p, str)] if isinstance(pin_corr, list | tuple) else []
    )
    feature_parameters: dict[str, Any] = {
        "include_timeframes": include_timeframes,
        "include_corr_pairlist": include_corr_pairlist,
        "label_period_candles": label_period_candles,
        "include_shifted_candles": 2,
        "indicator_periods_candles": list(indicator_periods_candles),
        # Remaining pins (DI_threshold, use_SVM_to_remove_outliers, an overriding
        # include_shifted_candles / indicator_periods_candles, …) win on overlap.
        **pin_features,
    }

    # Tunable model + DI/weight SLOTs live as class attributes on the (rendered)
    # template; read them off the class so the slot is the single source of truth
    # and the runtime config still carries the full research block.
    model_training_parameters: dict[str, Any] = {}
    if strategy_path is not None:
        di_threshold = extract_class_number(strategy_path, "DI_threshold")
        if di_threshold is not None:
            feature_parameters["DI_threshold"] = di_threshold
        weight_factor = extract_class_number(strategy_path, "weight_factor")
        if weight_factor is not None:
            feature_parameters["weight_factor"] = weight_factor
        model_training_parameters = extract_model_training_parameters(strategy_path)

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
        "model_training_parameters": model_training_parameters,
    }
