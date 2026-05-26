"""Stage 3 integration test — freqtrade subprocess + regime row.

BRD §13 Stage 3 DoD:

  - subprocess driver runs a 1-week backtest on cached BTC/USDT and parses
    the zip into a ``BacktestResult``,
  - REST client pings a dry-run instance (deferred: covered by Stage 7
    paper subgraph; this test focuses on the backtest path),
  - regime job writes a row.

This test verifies the first and third items end-to-end on real cached data
and a live Postgres ``app`` DB:

  1. Invoke :func:`orchestrator.tools.backtest_runner.run_backtest` with
     ``MeanReversionTemplate`` against 1 week of BTC/USDT 5m data already
     downloaded into ``freqtrade/user_data/data/binance/`` (Stage 3 commit
     3a populates this).
  2. Assert the returned ``BacktestResult`` has ``trades > 0`` — the v1
     baseline template MUST fire at least once on a week of liquid spot
     data. Zero trades on the default params means the template's entry
     logic is broken; this is the BRD-defined sanity floor.
  3. Read 30+ closes from the same cached feather, classify the regime,
     insert a ``regime_log`` row, and read it back to confirm persistence.

The test is opt-in via the ``integration`` + ``freqtrade`` markers (see
``pyproject.toml``) so the CI matrix job that runs ``pytest -m "not
integration"`` skips it. Local invocation:

    pytest -m "integration and freqtrade" -v tests/integration

Prerequisites (assumed by the test, will skip if not met):
  - ``docker`` available on PATH; ``freqtradeorg/freqtrade:stable_freqai``
    image pulled,
  - ``freqtrade/user_data/data/binance/BTC_USDT-5m.feather`` exists,
  - Postgres ``app`` DB reachable on ``DATABASE_URL`` (port 5433 per
    operator .env).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pyarrow.feather as feather
import pytest

from orchestrator.tools.backtest_runner import (
    SHARED_DATA_DIR,
    cleanup_worker,
    run_backtest,
)
from orchestrator.tools.regime import classify_regime, insert_regime_log

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STRATEGY_PATH = REPO_ROOT / "strategy_templates" / "mean_reversion_template.py"
CACHED_BTC_5M = SHARED_DATA_DIR / "binance" / "BTC_USDT-5m.feather"

# Operator-tuned via .env; falls back to the sqlalchemy-style URL stripped of
# its driver prefix because psycopg's async connector wants a plain libpq URI.
APP_DB_SQLALCHEMY_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://trading_agent:change_me_local_only@127.0.0.1:5433/app",
)
APP_DB_URI = APP_DB_SQLALCHEMY_URL.replace("postgresql+psycopg://", "postgresql://", 1)


def _skip_if_missing_prereqs() -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker not on PATH; skipping freqtrade subprocess integration test")
    if not CACHED_BTC_5M.exists():
        pytest.skip(
            f"cached BTC/USDT 5m feather missing at {CACHED_BTC_5M}; "
            "run the Stage 3a download-data command first"
        )
    # Confirm the image is locally available so we don't trigger an
    # unannounced ~1.5 GB pull inside the test.
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
@pytest.mark.freqtrade
@pytest.mark.asyncio
async def test_backtest_runner_produces_trades_on_cached_btc() -> None:
    """End-to-end backtest of the mean-reversion template on cached BTC/USDT 5m.

    Picks a 1-week window inside the 730-day cache that's known to contain
    market activity (not a holiday or thin-liquidity slice). The window
    chosen is the operator's stage-3 setup date minus 60 days, +1 week,
    which keeps the test deterministic against the cache as long as it's
    inside the downloaded range.
    """
    _skip_if_missing_prereqs()

    # 60–53 days before today, sliced to whole-day boundaries. The cache
    # contains 730 days, so this is well inside the available range.
    today = datetime.now(UTC).date()
    start = today.replace(day=1).fromordinal(today.toordinal() - 60)
    end = today.replace(day=1).fromordinal(today.toordinal() - 53)
    timerange = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"

    result = await run_backtest(
        STRATEGY_PATH,
        pairs=["BTC/USDT"],
        timeframe="5m",
        timerange=timerange,
        fold_id="stage3-smoke",
        param_set_id="defaults",
    )

    assert result["trades"] > 0, (
        f"mean-reversion template produced zero trades on {timerange}; "
        "either the template's entry logic is broken or the cached data slice is empty"
    )
    assert result["pair"] == "BTC/USDT"
    assert result["timeframe"] == "5m"
    assert result["fold_id"] == "stage3-smoke"
    assert result["raw_zip_path"], "BacktestResult.raw_zip_path is empty"
    assert Path(
        result["raw_zip_path"]
    ).exists(), f"BacktestResult.raw_zip_path points to a missing file: {result['raw_zip_path']}"

    # Tidy up the worker dir on success. We leave it on failure (the
    # assertions above would short-circuit cleanup) so the operator can
    # inspect the artifacts. This matches the docstring of cleanup_worker.
    worker_dir = Path(result["raw_zip_path"]).parent.parent
    cleanup_worker(worker_dir)


@pytest.mark.integration
@pytest.mark.freqtrade
@pytest.mark.asyncio
async def test_regime_classify_and_insert_writes_a_regime_log_row() -> None:
    """Classify a regime from real cached closes and persist it.

    Reads the most recent 200 closes from the cached BTC/USDT 5m feather,
    classifies via :func:`orchestrator.tools.regime.classify_regime`, inserts
    a ``regime_log`` row tagged with a per-test detector id (so the test is
    idempotent across re-runs), then reads the row back.
    """
    if not CACHED_BTC_5M.exists():
        pytest.skip(
            f"cached BTC/USDT 5m feather missing at {CACHED_BTC_5M}; "
            "run the Stage 3a download-data command first"
        )

    # pyarrow.feather is present via streamlit's deps; we use it directly
    # rather than pulling pandas through this test path. The pyarrow API is
    # not fully typed, hence the local ignore.
    table = feather.read_table(CACHED_BTC_5M)  # type: ignore[no-untyped-call]
    closes_full = table.column("close").to_pylist()
    closes = closes_full[-200:]  # last ~17h of 5m candles
    assert len(closes) >= 30

    regime, features = classify_regime(closes, timeframe_minutes=5)

    # Composite label must be one of the nine legal buckets. Checking the full
    # set is more robust than parsing the underscore form, since "low_vol" /
    # "mid_vol" / "high_vol" each contain their own underscore.
    legal_labels = {
        f"{v}_{t}" for v in ("low_vol", "mid_vol", "high_vol") for t in ("down", "flat", "up")
    }
    assert regime in legal_labels, f"unexpected regime label {regime!r}"

    detector_id = f"vol_trend_v1_test_{uuid.uuid4().hex[:8]}"

    async with await psycopg.AsyncConnection.connect(APP_DB_URI) as conn:
        await insert_regime_log(
            conn=conn,
            regime=regime,
            features=features,
            detector=detector_id,
        )
        await conn.commit()

        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT regime, features, detector FROM regime_log WHERE detector = %s",
                (detector_id,),
            )
            row = await cur.fetchone()

    assert row is not None, f"no regime_log row found for detector={detector_id}"
    persisted_regime, persisted_features, persisted_detector = row
    assert persisted_regime == regime
    assert persisted_detector == detector_id
    # features round-trip as JSONB → dict; psycopg's JSONB adapter returns dict.
    assert isinstance(persisted_features, dict)
    assert persisted_features["closes_used"] == features.closes_used
    assert persisted_features["timeframe_minutes"] == 5
