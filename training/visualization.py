"""Forecast visualization utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.model_io import safe_state_name


def plot_state_forecast(
    history_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    validation_predictions: np.ndarray,
    future_forecast_df: pd.DataFrame,
    state: str,
    model_name: str,
    output_dir: Path,
) -> Path:
    """Save a state-level plot with history, validation forecast, and future forecast."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{safe_state_name(state)}_forecast.png"

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(
        history_df["date"],
        history_df["sales"],
        label="Historical sales",
        color="#1f77b4",
        linewidth=2,
    )
    ax.plot(
        validation_df["date"],
        validation_predictions,
        label="Validation forecasts",
        color="#ff7f0e",
        linewidth=2,
        linestyle="--",
    )
    ax.plot(
        future_forecast_df["date"],
        future_forecast_df["predicted_sales"],
        label="Predicted sales",
        color="#2ca02c",
        linewidth=2,
    )

    ax.axvline(
        validation_df["date"].min(),
        color="#666666",
        linestyle=":",
        linewidth=1,
        label="Validation start",
    )
    ax.axvline(
        future_forecast_df["date"].min(),
        color="#222222",
        linestyle=":",
        linewidth=1,
        label="Forecast start",
    )
    ax.set_title(f"{state} Sales Forecast ({model_name})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path
