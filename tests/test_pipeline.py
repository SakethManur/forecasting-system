"""Tests for the train/validation split, baseline and state validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.pipeline import (
    _improvement_pct,
    _naive_baseline,
    _time_series_split,
    _validate_state_history,
    _validation_periods,
)
from utils.config import settings
from utils.exceptions import DataValidationError


def _weekly_frame(observed: list[bool], sales=None) -> pd.DataFrame:
    n = len(observed)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-02", periods=n, freq="W-SUN"),
            "state": "S",
            "sales": np.arange(1, n + 1, dtype=float) * 10 if sales is None else sales,
            "holiday_flag": 0,
            "is_observed": observed,
        }
    )
    frame.attrs.update({"frequency": "W-SUN", "step_days": 7})
    return frame


def test_validation_window_is_the_default_when_data_is_dense():
    frame = _weekly_frame([True] * 60)
    assert _validation_periods(frame) == int(np.ceil(settings.validation_days / 7))


def test_validation_window_extends_until_it_holds_enough_real_weeks():
    # Only every third week is real in the last part of the series.
    observed = [True] * 40 + [i % 3 == 0 for i in range(30)]
    frame = _weekly_frame(observed)

    periods = _validation_periods(frame)
    window = frame.iloc[-periods:]

    assert periods > int(np.ceil(settings.validation_days / 7))
    assert int(window["is_observed"].sum()) == settings.min_validation_observations
    assert window["is_observed"].iloc[0]  # the window starts on a real week


def test_split_keeps_order_sizes_and_attrs():
    frame = _weekly_frame([True] * 60)
    train_df, val_df = _time_series_split(frame)

    assert len(train_df) + len(val_df) == len(frame)
    assert train_df["date"].max() < val_df["date"].min()
    assert val_df.attrs["step_days"] == 7
    assert train_df.attrs["frequency"] == "W-SUN"


def test_validate_rejects_a_series_too_short_to_train():
    with pytest.raises(DataValidationError, match="required for training"):
        _validate_state_history(_weekly_frame([True] * 15), "Tiny")


def test_validate_rejects_a_validation_window_without_real_data():
    observed = [True] * 5 + [False] * 30 + [False] * 8
    with pytest.raises(DataValidationError, match="no real observations"):
        _validate_state_history(_weekly_frame(observed), "Sparse")


def test_validate_accepts_a_healthy_series():
    _validate_state_history(_weekly_frame([True] * 60), "Healthy")


def test_naive_baseline_repeats_the_last_real_training_value():
    train_df = _weekly_frame([True, True, False], sales=[100.0, 120.0, 130.0])
    val_df = pd.DataFrame({"sales": [110.0, 140.0], "is_observed": [True, True]})

    baseline = _naive_baseline(train_df, val_df)

    # 120 is the last real value; 130 was interpolated and must be skipped.
    assert baseline["method"] == "naive_last_observed_value"
    assert baseline["metrics"]["mae"] == pytest.approx(15.0)


@pytest.mark.parametrize(
    ("baseline_rmse", "model_rmse", "expected"),
    [(100.0, 78.0, 22.0), (100.0, 120.0, -20.0), (0.0, 5.0, None)],
)
def test_improvement_pct(baseline_rmse, model_rmse, expected):
    result = _improvement_pct(baseline_rmse, model_rmse)
    assert result == (pytest.approx(expected) if expected is not None else None)
