"""Stage 4c unit tests — anchored walk-forward planner (BRD §5.4).

BRD §5.4 spec: "anchored 6-fold walk-forward (4 months train / 1 month
test, sliding by 1 month)". Decomposed:

  - fold 1: train [data_start, +4mo), test [+4mo, +5mo)
  - fold 2: train [data_start, +5mo), test [+5mo, +6mo)
  - ...
  - fold 6: train [data_start, +9mo), test [+9mo, +10mo)

The anchored property is the train START stays fixed; only the train end
and test window slide. Rolling walk-forward (fixed train size, both ends
slide) is supported via ``anchored=False`` for future comparison but is
not the BRD-mandated v1 default.

These tests are offline — no Docker, no Postgres, no Freqtrade. They
verify the timerange arithmetic only.
"""

from __future__ import annotations

from datetime import date

import pytest

from orchestrator.subgraphs.validation import plan_walk_forward


def test_default_six_fold_anchored_walk_forward_from_jan_1_2024() -> None:
    folds = plan_walk_forward(data_start=date(2024, 1, 1))
    assert len(folds) == 6
    # Fold 1: train Jan 1 → May 1; test May 1 → Jun 1.
    assert folds[0] == {
        "fold_id": "fold_1",
        "timerange": "20240501-20240601",
        "train_timerange": "20240101-20240501",
    }
    # Fold 6: train Jan 1 → Oct 1; test Oct 1 → Nov 1.
    assert folds[5] == {
        "fold_id": "fold_6",
        "timerange": "20241001-20241101",
        "train_timerange": "20240101-20241001",
    }


def test_anchored_train_starts_are_all_equal_to_data_start() -> None:
    """Anchored = train_start never moves. Catches off-by-one drift."""
    folds = plan_walk_forward(data_start=date(2024, 3, 1))
    train_starts = {f["train_timerange"].split("-")[0] for f in folds}
    assert train_starts == {"20240301"}, (
        f"Anchored walk-forward must keep train start fixed at data_start. "
        f"Got distinct train starts: {train_starts}"
    )


def test_test_windows_tile_without_overlap_or_gap() -> None:
    """Each fold's test window starts exactly where the previous one ended."""
    folds = plan_walk_forward(data_start=date(2024, 1, 1))
    for prev, curr in zip(folds, folds[1:], strict=False):
        prev_end = prev["timerange"].split("-")[1]
        curr_start = curr["timerange"].split("-")[0]
        assert prev_end == curr_start, (
            f"Adjacent test windows must tile: previous fold ended {prev_end}, "
            f"current fold starts {curr_start}"
        )


def test_rolling_mode_slides_train_start_too() -> None:
    """anchored=False: both train_start and test_start advance by 1 month."""
    folds = plan_walk_forward(data_start=date(2024, 1, 1), anchored=False)
    assert len(folds) == 6
    assert folds[0]["train_timerange"] == "20240101-20240501"
    assert folds[1]["train_timerange"] == "20240201-20240601"
    assert folds[5]["train_timerange"] == "20240601-20241001"


def test_custom_fold_count_and_window_sizes() -> None:
    folds = plan_walk_forward(
        data_start=date(2024, 1, 1),
        train_months=3,
        test_months=2,
        n_folds=4,
    )
    assert len(folds) == 4
    # Fold 1: train Jan→Apr (3mo), test Apr→Jun (2mo).
    assert folds[0]["train_timerange"] == "20240101-20240401"
    assert folds[0]["timerange"] == "20240401-20240601"
    # Test windows slide by test_months, train end slides by 1 month.
    assert folds[1]["train_timerange"] == "20240101-20240501"
    assert folds[1]["timerange"] == "20240501-20240701"


def test_end_of_month_clamping_handles_31_jan_plus_1_month() -> None:
    """31 Jan + 1 month → 28/29 Feb. The helper must not raise."""
    folds = plan_walk_forward(data_start=date(2024, 1, 31), n_folds=1, train_months=1)
    # 31 Jan + 1 month = 29 Feb 2024 (leap year). Test window is 1 month
    # after that = 29 Mar 2024.
    assert folds[0]["train_timerange"].startswith("20240131-")
    assert folds[0]["train_timerange"].endswith("-20240229")


def test_invalid_params_raise_with_useful_messages() -> None:
    with pytest.raises(ValueError, match="n_folds"):
        plan_walk_forward(data_start=date(2024, 1, 1), n_folds=0)
    with pytest.raises(ValueError, match="train_months"):
        plan_walk_forward(data_start=date(2024, 1, 1), train_months=0)
    with pytest.raises(ValueError, match="test_months"):
        plan_walk_forward(data_start=date(2024, 1, 1), test_months=0)
