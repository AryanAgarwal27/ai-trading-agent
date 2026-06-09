"""Stage 8i unit tests — walk-forward anchored to the real cache range.

BRD §5.4 mandates an anchored 6-fold walk-forward. Stage 7h anchored it to
``today - 365d`` (a heuristic in ``_default_walk_forward_start``), which can
emit OOS timeranges that run past the freshest cached candle — producing
zero-trade folds that the Stage 7h ``gate_backtest`` per-fold guard then
hard-fails. Stage 8i replaces the heuristic with a reader of the cached
feather's actual ``date`` column so the whole window provably sits inside
the data on disk.

These tests are offline — they synthesise tiny fixture feathers (just a
``date`` column spanning known dates) in ``tmp_path``; no Docker, no
Postgres, no real OHLCV download.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.feather as feather
import pytest

from orchestrator.subgraphs import validation as wf
from orchestrator.subgraphs.validation import (
    _default_walk_forward_start,
    check_walk_forward_window_fits,
    prepare_validation_inputs,
    read_feather_date_range,
    walk_forward_from_cache,
)


def _make_feather(path: Path, start: date, end: date) -> None:
    """Write a minimal Freqtrade-shaped feather spanning ``[start, end]``.

    Only the ``date`` column matters for the range reader; a ``close``
    column is included so the fixture mirrors a real feather's shape.
    The ``date`` column is ``timestamp[ms, tz=UTC]`` — identical dtype to
    a real Freqtrade OHLCV feather.
    """
    stamps = [
        datetime(start.year, start.month, start.day, tzinfo=UTC),
        datetime(end.year, end.month, end.day, tzinfo=UTC),
    ]
    table = pa.table(
        {
            "date": pa.array(stamps, type=pa.timestamp("ms", tz="UTC")),
            "close": pa.array([1.0, 2.0]),
        }
    )
    feather.write_feather(table, path)  # type: ignore[no-untyped-call]


def _bounds(folds: list[dict[str, str]]) -> tuple[date, date]:
    """Earliest train_start and latest test_end across the folds (as dates)."""
    starts = [datetime.strptime(f["train_timerange"].split("-")[0], "%Y%m%d").date() for f in folds]
    ends = [datetime.strptime(f["timerange"].split("-")[1], "%Y%m%d").date() for f in folds]
    return min(starts), max(ends)


# ─── read_feather_date_range ───────────────────────────────────────────


def test_read_feather_date_range_returns_min_and_max(tmp_path: Path) -> None:
    path = tmp_path / "BTC_USDT-5m.feather"
    _make_feather(path, date(2024, 1, 1), date(2024, 12, 1))
    lo, hi = read_feather_date_range(path)
    assert lo == date(2024, 1, 1)
    assert hi == date(2024, 12, 1)


def test_read_feather_date_range_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_feather_date_range(tmp_path / "nope.feather")


# ─── walk_forward_from_cache — happy path ──────────────────────────────


def test_six_folds_all_inside_a_full_cache_span(tmp_path: Path) -> None:
    """A cache comfortably larger than the 10-month window → 6 folds, all
    timeranges inside [min_date, max_date]."""
    path = tmp_path / "BTC_USDT-5m.feather"
    _make_feather(path, date(2024, 1, 1), date(2024, 12, 1))

    folds = walk_forward_from_cache(path)

    assert len(folds) == 6
    lo, hi = _bounds(folds)
    assert lo >= date(2024, 1, 1), f"earliest train_start {lo} precedes cache min"
    assert hi <= date(2024, 12, 1), f"latest test_end {hi} runs past cache max"


def test_last_oos_fold_ends_at_cache_max(tmp_path: Path) -> None:
    """The window is anchored so the final OOS fold's test_end == cache max,
    using the freshest available data without overrunning it."""
    path = tmp_path / "BTC_USDT-5m.feather"
    _make_feather(path, date(2024, 1, 1), date(2024, 12, 1))

    folds = walk_forward_from_cache(path)

    last_test_end = folds[-1]["timerange"].split("-")[1]
    assert last_test_end == "20241201"


# ─── walk_forward_from_cache — graceful degradation ────────────────────


def test_short_cache_degrades_to_fewer_folds(tmp_path: Path) -> None:
    """A 7-month cache can't fit the full 10-month/6-fold window; it
    degrades to fewer folds rather than emitting out-of-range timeranges."""
    path = tmp_path / "BTC_USDT-5m.feather"
    _make_feather(path, date(2024, 1, 1), date(2024, 8, 1))

    folds = walk_forward_from_cache(path)

    assert 1 <= len(folds) < 6
    lo, hi = _bounds(folds)
    assert lo >= date(2024, 1, 1)
    assert hi <= date(2024, 8, 1)


def test_cache_too_short_for_one_fold_raises_explicitly(tmp_path: Path) -> None:
    """A cache shorter than one train+test window (5 months) can't produce a
    single valid fold — raise an explicit error, never out-of-range folds."""
    path = tmp_path / "BTC_USDT-5m.feather"
    _make_feather(path, date(2024, 1, 1), date(2024, 4, 1))  # 3 months

    with pytest.raises(ValueError, match="too short"):
        walk_forward_from_cache(path)


# ─── Stage 13: configurable walk-forward window (data_start override) ───────


def _cache_a_feather(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lo: date, hi: date) -> None:
    """Place a BTC/USDT 15m feather where _anchor_feather_path looks for it."""
    binance = tmp_path / "binance"
    binance.mkdir()
    _make_feather(binance / "BTC_USDT-15m.feather", lo, hi)
    monkeypatch.setattr(wf, "SHARED_DATA_DIR", tmp_path)


def test_window_fits_returns_none_for_in_range_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit data_start whose 10mo/6-fold window sits inside the cache → OK."""
    _cache_a_feather(tmp_path, monkeypatch, date(2024, 1, 1), date(2025, 6, 1))
    # 2024-02-01 + 10mo = 2024-12-01, inside [2024-01-01, 2025-06-01].
    assert check_walk_forward_window_fits(["BTC/USDT"], "15m", date(2024, 2, 1)) is None


def test_window_before_cache_start_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cache_a_feather(tmp_path, monkeypatch, date(2024, 1, 1), date(2025, 6, 1))
    err = check_walk_forward_window_fits(["BTC/USDT"], "15m", date(2023, 1, 1))
    assert err is not None
    assert "before the cached range start" in err


def test_window_past_cache_end_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A data_start so late the 10-month window overruns max_date → clear error."""
    _cache_a_feather(tmp_path, monkeypatch, date(2024, 1, 1), date(2024, 12, 1))
    err = check_walk_forward_window_fits(["BTC/USDT"], "15m", date(2024, 9, 1))
    assert err is not None
    assert "runs past the cached range end" in err


def test_window_check_no_cached_feather_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wf, "SHARED_DATA_DIR", tmp_path)  # empty → no feather
    err = check_walk_forward_window_fits(["BTC/USDT"], "15m", date(2024, 2, 1))
    assert err is not None
    assert "no cached OHLCV" in err


def test_window_check_fewer_folds_fit_a_short_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A window that overruns at 6 folds (10mo) fits at 3 folds (7mo) from an
    earlier start — the operator can dial n_folds down to fit a shorter span."""
    _cache_a_feather(tmp_path, monkeypatch, date(2024, 1, 1), date(2024, 12, 1))
    # 6 folds (10mo) from 2024-09-01 → ends 2025-07-01, past the 2024-12-01 max.
    assert check_walk_forward_window_fits(["BTC/USDT"], "15m", date(2024, 9, 1)) is not None
    # 3 folds (7mo) from 2024-05-01 → ends exactly 2024-12-01 (== max) → fits.
    assert check_walk_forward_window_fits(["BTC/USDT"], "15m", date(2024, 5, 1), n_folds=3) is None


# ─── prepare_validation_inputs honors / ignores the override ────────────────


def test_prepare_validation_inputs_honors_data_start_override() -> None:
    """An explicit data_start in artifacts anchors the folds there (Stage 13)."""
    state: dict[str, Any] = {
        "artifacts": {"walk_forward_override": {"data_start": "2024-09-01", "n_folds": 6}},
    }
    out = prepare_validation_inputs(state)  # type: ignore[arg-type]
    folds = out["folds"]
    assert len(folds) == 6
    # Anchored: every fold's train_start == data_start; fold 1 train_start too.
    assert folds[0]["train_timerange"].split("-")[0] == "20240901"
    assert folds[-1]["train_timerange"].split("-")[0] == "20240901"


def test_prepare_validation_inputs_override_n_folds() -> None:
    state: dict[str, Any] = {
        "artifacts": {"walk_forward_override": {"data_start": "2024-09-01", "n_folds": 3}},
    }
    out = prepare_validation_inputs(state)  # type: ignore[arg-type]
    assert len(out["folds"]) == 3


def test_prepare_validation_inputs_default_window_when_no_override() -> None:
    """No override + no resolvable feather → byte-identical today-365 default."""
    state: dict[str, Any] = {"artifacts": {}}  # no override, no pairs/tf → heuristic fallback
    out = prepare_validation_inputs(state)  # type: ignore[arg-type]
    folds = out["folds"]
    assert len(folds) == 6
    expected_start = _default_walk_forward_start().strftime("%Y%m%d")
    assert folds[0]["train_timerange"].split("-")[0] == expected_start
