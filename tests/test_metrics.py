"""Tests for evaluation metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from training.metrics import regression_metrics, validation_metrics


def test_regression_metrics_values():
    metrics = regression_metrics([100.0, 200.0], [110.0, 180.0])

    assert metrics["mae"] == pytest.approx(15.0)
    assert metrics["rmse"] == pytest.approx(math.sqrt((10**2 + 20**2) / 2))
    assert metrics["mape"] == pytest.approx(10.0)  # (10% + 10%) / 2


def test_mape_skips_zero_actuals():
    assert regression_metrics([0.0, 100.0], [5.0, 90.0])["mape"] == pytest.approx(10.0)


def test_mape_is_nan_when_every_actual_is_zero():
    assert math.isnan(regression_metrics([0.0, 0.0], [1.0, 2.0])["mape"])


def test_validation_metrics_ignore_interpolated_periods():
    # Regression test: interpolated weeks used to be scored, so a model that
    # reproduced the straight interpolated line looked near-perfect.
    val_df = pd.DataFrame(
        {
            "sales": [100.0, 150.0, 200.0, 250.0],
            "is_observed": [True, False, False, True],
        }
    )
    predictions = np.array([110.0, 150.0, 200.0, 240.0])

    metrics = validation_metrics(val_df, predictions)

    assert metrics["rmse"] == pytest.approx(10.0)  # only the two real weeks count
    assert metrics["mae"] == pytest.approx(10.0)


def test_validation_metrics_use_every_row_without_an_observed_flag():
    val_df = pd.DataFrame({"sales": [100.0, 200.0]})
    assert validation_metrics(val_df, [100.0, 180.0])["mae"] == pytest.approx(10.0)


def test_validation_metrics_refuse_a_window_with_no_real_data():
    val_df = pd.DataFrame({"sales": [1.0, 2.0], "is_observed": [False, False]})
    with pytest.raises(ValueError, match="no observed periods"):
        validation_metrics(val_df, [1.0, 2.0])
