"""Model comparison reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def generate_model_comparison_report(
    registry: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Save per-state model comparison CSV/JSON and ranked model summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for state, metadata in sorted(registry.get("states", {}).items()):
        selected_best_model = metadata["best_model"]
        for model_name, model_info in sorted(metadata.get("models", {}).items()):
            metrics = model_info.get("metrics") or {}
            rows.append(
                {
                    "state": state,
                    "model_name": model_name,
                    "RMSE": metrics.get("rmse"),
                    "MAE": metrics.get("mae"),
                    "training_time": model_info.get("training_time"),
                    "selected_best_model": selected_best_model,
                    "status": model_info.get("status"),
                }
            )

    comparison_df = pd.DataFrame(rows)
    csv_path = output_dir / "model_comparison.csv"
    json_path = output_dir / "model_comparison.json"
    comparison_df.to_csv(csv_path, index=False)

    ranked_summary = _rank_best_models(comparison_df)
    payload = {
        "model_comparison": comparison_df.where(pd.notna(comparison_df), None).to_dict(
            orient="records"
        ),
        "ranked_summary": ranked_summary,
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "ranked_summary": ranked_summary,
    }


def _rank_best_models(comparison_df: pd.DataFrame) -> list[dict[str, Any]]:
    if comparison_df.empty:
        return []

    trained = comparison_df[comparison_df["status"] == "trained"].copy()
    if trained.empty:
        return []

    trained["best_model_win"] = (
        trained["model_name"] == trained["selected_best_model"]
    ).astype(int)
    summary = (
        trained.groupby("model_name", as_index=False)
        .agg(
            states_evaluated=("state", "nunique"),
            best_model_wins=("best_model_win", "sum"),
            average_RMSE=("RMSE", "mean"),
            average_MAE=("MAE", "mean"),
            average_training_time=("training_time", "mean"),
        )
        .sort_values(
            ["best_model_wins", "average_RMSE", "average_MAE"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    summary["rank"] = summary.index + 1
    columns = [
        "rank",
        "model_name",
        "states_evaluated",
        "best_model_wins",
        "average_RMSE",
        "average_MAE",
        "average_training_time",
    ]
    return summary[columns].where(pd.notna(summary[columns]), None).to_dict(
        orient="records"
    )
