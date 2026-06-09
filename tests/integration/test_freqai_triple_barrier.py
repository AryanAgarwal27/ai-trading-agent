"""Stage 12 integration test — the FreqAI triple-barrier CLASSIFIER end-to-end.

Acceptance (operator requirement #6/#7): one fold runs train → backtest → gate,
proving the **LightGBMClassifier** trains and the ``up`` P(up) probability column
is read correctly by the entry logic. (The column is ``up`` — the class value —
not ``&-trade_up_proba``: freqtrade 2026.4's ``BaseClassifierModel.predict``
names probability columns after ``model.classes_`` and appends them unprefixed.
This test caught that exact mistake in the first draft.)

How the proof works (no dataframe is exposed by ``run_backtest``, so the proof is
structural):

  - ``freqai_model_for`` keyed off the template's class name resolves
    ``LightGBMClassifier`` — the classifier path, NOT the regressor default
    (requirement #6). Asserted deterministically (no Docker needed for this leg).
  - A REAL ``run_backtest`` on the template completes and returns a
    ``BacktestResult``. Reaching that point means: FreqAI was enabled, the
    LightGBM **classifier** TRAINED on the triple-barrier ``&-trade`` target, and
    ``populate_entry_trend`` evaluated ``dataframe["up"] > entry_proba`` over
    every row WITHOUT a KeyError. If the probability column were named wrong for
    this FreqAI version, that lookup would raise inside the
    container → Freqtrade exits non-zero → ``run_backtest`` raises
    ``BacktestError`` and this test FAILS. So a clean ``BacktestResult`` IS the
    "column read correctly" proof — the canary the research §D asked for.

Zero trades is a VALID outcome (the meta-gate is high-conviction by design) and
does NOT weaken the proof: ``populate_entry_trend`` references the column for the
whole frame regardless of whether any row passes the gate, so a missing column
fails fast no matter the trade count. We assert completion + metric shape, never
profitability.

The template's ``freqai_config`` pins ``include_timeframes ["5m","1h"]`` +
``include_corr_pairlist ["BTC/USDT","ETH/USDT"]``, so FreqAI needs BTC **and**
ETH data at BOTH 5m and 1h cached — the skip-guard lists exactly what to
download.

Opt-in via the ``integration`` + ``freqtrade`` markers (skipped by CI's
``pytest -m "integration and not freqtrade"``). Local:

    pytest -m "integration and freqtrade" -v tests/integration/test_freqai_triple_barrier.py
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from orchestrator.tools.backtest_runner import (
    SHARED_DATA_DIR,
    _extract_strategy_class_name,
    cleanup_worker,
    run_backtest,
)
from orchestrator.tools.freqai_config import freqai_model_for

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRIPLE_BARRIER_PATH = REPO_ROOT / "strategy_templates" / "freqai_triple_barrier_template.py"

# The template pins 5m+1h timeframes and BTC/ETH corr pairs (research §C), so
# FreqAI needs all four feathers present.
_REQUIRED_FEATHERS = [
    SHARED_DATA_DIR / "binance" / "BTC_USDT-5m.feather",
    SHARED_DATA_DIR / "binance" / "BTC_USDT-1h.feather",
    SHARED_DATA_DIR / "binance" / "ETH_USDT-5m.feather",
    SHARED_DATA_DIR / "binance" / "ETH_USDT-1h.feather",
]


def _skip_if_missing_prereqs() -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker not on PATH; skipping triple-barrier freqtrade integration test")
    missing = [f for f in _REQUIRED_FEATHERS if not f.exists()]
    if missing:
        names = ", ".join(f.name for f in missing)
        pytest.skip(
            f"cached feather(s) missing: {names}. The triple-barrier template uses "
            "5m+1h timeframes and BTC/ETH corr pairs; download with "
            "`download-data --exchange binance --pairs BTC/USDT ETH/USDT "
            "--timeframes 5m 1h --days 730`"
        )
    result = subprocess.run(
        ["docker", "image", "inspect", "freqtradeorg/freqtrade:stable_freqai"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(
            "freqtradeorg/freqtrade:stable_freqai image not found locally; "
            "run `docker pull freqtradeorg/freqtrade:stable_freqai` first"
        )


@pytest.mark.integration
def test_triple_barrier_resolves_to_lightgbm_classifier() -> None:
    """Requirement #6: the template's class resolves to the CLASSIFIER model.

    Deterministic (no Docker, no @freqtrade) so it runs in the CI integration job
    (``-m "integration and not freqtrade"``) — derives the strategy class name
    from the file the same way the runner does, then checks ``freqai_model_for``
    keys off it. Stronger than the unit check: catches a class-name typo in the
    actual template file, not just a hardcoded string.
    """
    class_name = _extract_strategy_class_name(TRIPLE_BARRIER_PATH)
    assert class_name == "FreqaiTripleBarrierClassifier"
    assert freqai_model_for(class_name) == "LightGBMClassifier"


@pytest.mark.integration
@pytest.mark.freqtrade
@pytest.mark.asyncio
async def test_triple_barrier_classifier_trains_and_reads_up_proba(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One fold: the LightGBMClassifier trains and the entry logic reads P(up).

    See the module docstring for why a clean ``BacktestResult`` proves both the
    classifier trained AND ``up`` (P(up)) was read by ``populate_entry_trend`` (a
    wrong column name would raise ``BacktestError``). A losing / zero-trade
    outcome is a valid gauntlet result, not a setup error — we assert completion
    + metric shape, not profitability.
    """
    _skip_if_missing_prereqs()

    # A ~40-day backtest window ~200 days back: deep mid-cache so FreqAI's
    # train_period_days=30 runway (auto-loaded BEFORE --timerange) sits entirely
    # on cached data, and clear of the freshest (possibly partial) candles.
    today = datetime.now(UTC).date()
    start = today.fromordinal(today.toordinal() - 200)
    end = today.fromordinal(today.toordinal() - 160)
    timerange = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"

    result = await run_backtest(
        TRIPLE_BARRIER_PATH,
        pairs=["BTC/USDT"],
        timeframe="5m",
        timerange=timerange,
        fold_id="tb-classifier-smoke",
        param_set_id="defaults",
        # Classifier + 50-80 features × 5m/1h × corr pairs trains slower than the
        # regressor; give it generous headroom so a slow train is not a timeout.
        timeout_s=1200,
    )

    # Completion + metric shape. Reaching here at all means FreqAI was enabled,
    # the LightGBMClassifier trained on the &-trade target, and the entry logic
    # read the "up" P(up) column without a KeyError (the column-name canary).
    assert result["fold_id"] == "tb-classifier-smoke"
    assert result["pair"] == "BTC/USDT"
    assert result["timeframe"] == "5m"
    assert isinstance(result["trades"], int) and result["trades"] >= 0
    assert isinstance(result["is_sharpe"], float)
    assert isinstance(result["profit_factor"], float)
    assert isinstance(result["max_dd"], float)
    assert result["raw_zip_path"] and Path(result["raw_zip_path"]).exists()

    print(
        f"\ntriple-barrier classifier {timerange}: trades={result['trades']} "
        f"sharpe={result['is_sharpe']:.3f} pf={result['profit_factor']:.3f} "
        f"max_dd={result['max_dd']:.3f}"
    )
    capsys.readouterr()

    cleanup_worker(Path(result["raw_zip_path"]).parent.parent)
