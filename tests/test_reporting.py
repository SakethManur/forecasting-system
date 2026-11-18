"""Tests for the model comparison report."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from training.reporting import generate_model_comparison_report


def _registry():
    def state(best, sarima_rmse, xgb_rmse, baseline_rmse, improvement):
        return {
            "best_model": best,
            "models": {
                "sarima": {"status": "trained", "metrics": {"rmse": sarima_rmse, "mae": 1.0, "mape": 2.0}, "training_time": 0.1},
                "xgboost": {"status": "trained", "metrics": {"rmse": xgb_rmse, "mae": 1.0, "mape": 2.0}, "training_time": 0.2},
                "lstm": {"status": "failed", "error": "boom"},
            },
            "baseline": {"method": "naive_last_observed_value", "metrics": {"rmse": baseline_rmse, "mae": 2.0, "mape": 3.0}},
            "rmse_improvement_vs_baseline_pct": improvement,
        }

    return {
        "states": {
            "A": state("xgboost", 12.0, 8.0, 10.0, 20.0),
            "B": state("sarima", 6.0, 9.0, 10.0, 40.0),
            "C": state("sarima", 11.0, 12.0, 10.0, -10.0),
        }
    }


def test_report_writes_csv_and_json_with_baseline_rows(tmp_path):
    report = generate_model_comparison_report(_registry(), tmp_path)

    csv = pd.read_csv(report["csv_path"])
    assert set(csv["status"]) == {"trained", "failed", "baseline"}
    assert (csv["model_name"] == "naive_last_observed_value").sum() == 3
    assert "MAPE" in csv.columns

    payload = json.loads((tmp_path / "model_comparison.json").read_text())
    assert payload["baseline_summary"] == report["baseline_summary"]


def test_ranking_counts_wins_and_never_includes_the_baseline(tmp_path):
    ranked = generate_model_comparison_report(_registry(), tmp_path)["ranked_summary"]

    names = [row["model_name"] for row in ranked]
    assert names == ["sarima", "xgboost"]
    assert ranked[0]["best_model_wins"] == 2
    assert ranked[0]["states_evaluated"] == 3


def test_baseline_summary_reports_improvement_across_states(tmp_path):
    summary = generate_model_comparison_report(_registry(), tmp_path)["baseline_summary"]

    assert summary["states_compared"] == 3
    assert summary["states_where_best_model_beats_baseline"] == 2
    assert summary["mean_rmse_improvement_pct"] == pytest.approx(50.0 / 3)
    assert summary["median_rmse_improvement_pct"] == pytest.approx(20.0)


def test_empty_registry_produces_an_empty_report(tmp_path):
    report = generate_model_comparison_report({"states": {}}, tmp_path)
    assert report["ranked_summary"] == []
    assert report["baseline_summary"] == {}
