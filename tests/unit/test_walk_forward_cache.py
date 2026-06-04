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

import pyarrow as pa
import pyarrow.feather as feather
import pytest

from orchestrator.subgraphs.validation import (
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
