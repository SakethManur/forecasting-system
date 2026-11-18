"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    nonzero = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
        if nonzero.any()
        else float("nan")
    )
    return {"rmse": rmse, "mae": mae, "mape": mape}


def validation_metrics(val_df: pd.DataFrame, predictions) -> dict[str, float]:
    """Score predictions against real observations only.

    Periods filled by interpolation are not real sales, so they are excluded;
    scoring them would reward models for reproducing the interpolation.
    """

    y_true = val_df["sales"].to_numpy(dtype=float)
    y_pred = np.asarray(predictions, dtype=float)
    if "is_observed" in val_df.columns:
        observed = val_df["is_observed"].to_numpy(dtype=bool)
        y_true, y_pred = y_true[observed], y_pred[observed]
    if len(y_true) == 0:
        raise ValueError("Validation window has no observed periods to score.")
    return regression_metrics(y_true, y_pred)
