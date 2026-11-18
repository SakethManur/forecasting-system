"""End-to-end test: train all four model families, then forecast.

Uses a small synthetic weekly dataset (see conftest.make_weekly_sales) and
temporary output folders, so it runs in well under a minute.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from training.pipeline import MODEL_NAMES, _predict_with_model, list_models, predict_state
from utils.config import settings
from utils.model_io import load_history


def test_every_state_trains_and_selects_a_best_model(trained_registry):
    assert sorted(trained_registry["states"]) == ["Alpha", "Beta"]
    assert trained_registry["failed_states"] == {}

    for metadata in trained_registry["states"].values():
        trained = {name for name, info in metadata["models"].items() if info["status"] == "trained"}
        assert trained == set(MODEL_NAMES)
        assert metadata["best_model"] in trained
        best_rmse = metadata["best_metrics"]["rmse"]
        assert best_rmse == min(metadata["models"][name]["metrics"]["rmse"] for name in trained)


def test_models_are_scored_on_real_weeks_against_a_baseline(trained_registry):
    for metadata in trained_registry["states"].values():
        assert metadata["validation_observations"] >= settings.min_validation_observations
        assert metadata["validation_periods"] >= metadata["validation_observations"]
        assert metadata["baseline"]["method"] == "naive_last_observed_value"
        assert metadata["rmse_improvement_vs_baseline_pct"] is not None
        # No suspiciously perfect scores like the old interpolation-scoring bug produced.
        assert metadata["best_metrics"]["rmse"] > 1_000


def test_comparison_report_is_written(trained_registry):
    report = trained_registry["model_comparison_report"]
    payload = json.loads(open(report["json_path"], encoding="utf-8").read())

    assert payload["baseline_summary"]["states_compared"] == 2
    assert {row["model_name"] for row in payload["ranked_summary"]} == set(MODEL_NAMES)


def test_registry_is_saved_and_listed(trained_registry):
    assert sorted(list_models()["states"]) == ["Alpha", "Beta"]


def test_predict_state_returns_future_weekly_forecasts(trained_registry):
    history = load_history("Alpha")
    forecast = predict_state("Alpha", horizon_days=28)

    assert len(forecast) == 4
    assert forecast["date"].min() > history["date"].max()
    assert forecast["date"].diff().dropna().dt.days.eq(7).all()
    assert (forecast["predicted_sales"] >= 0).all()


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_each_model_forecasts_at_the_series_cadence(trained_registry, model_name):
    # Regression test for Prophet, which used to forecast daily steps on weekly data.
    metadata = trained_registry["states"]["Beta"]
    history = load_history("Beta")

    forecast = _predict_with_model("Beta", model_name, history, 21, metadata)

    assert len(forecast) == 3
    assert forecast["date"].diff().dropna().dt.days.eq(7).all()
    assert np.isfinite(forecast["predicted_sales"]).all()
    # Forecasts should stay in the same range as the series (Beta is ~2M a week).
    assert forecast["predicted_sales"].between(500_000, 5_000_000).all()


def test_unknown_state_is_rejected(trained_registry):
    from utils.exceptions import ForecastingError

    with pytest.raises(ForecastingError, match="Unknown state"):
        predict_state("Atlantis")
