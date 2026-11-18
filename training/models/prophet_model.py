"""Prophet training and forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from training.metrics import validation_metrics
from training.models.common import ModelResult
from utils.model_io import save_joblib_model


MODEL_NAME = "prophet"


def train_prophet(train_df: pd.DataFrame, val_df: pd.DataFrame, state: str) -> ModelResult:
    from prophet import Prophet

    frequency = str(train_df.attrs.get("frequency", "D"))
    step_days = int(train_df.attrs.get("step_days", 1))
    # Prophet handles gaps natively, so fit on real observations only.
    observed_train = (
        train_df[train_df["is_observed"]] if "is_observed" in train_df.columns else train_df
    )
    prophet_train = observed_train.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]]
    hyperparameters = {
        "yearly_seasonality": True,
        # A day-of-week pattern only exists in daily data.
        "weekly_seasonality": step_days == 1,
        "daily_seasonality": False,
        "interval_width": 0.95,
        "forecast_frequency": frequency,
    }
    model = Prophet(
        yearly_seasonality=hyperparameters["yearly_seasonality"],
        weekly_seasonality=hyperparameters["weekly_seasonality"],
        daily_seasonality=hyperparameters["daily_seasonality"],
        interval_width=hyperparameters["interval_width"],
    )
    model.fit(prophet_train)

    # Forecast exactly the validation dates (the series' own cadence, not daily).
    forecast = model.predict(val_df[["date"]].rename(columns={"date": "ds"}))
    predictions = forecast["yhat"].to_numpy()
    metrics = validation_metrics(val_df, predictions)
    artifact_path = save_joblib_model(model, state, MODEL_NAME)
    return ModelResult(MODEL_NAME, predictions, metrics, artifact_path, hyperparameters)


def predict_prophet(model, history: pd.DataFrame, horizon: int, frequency: str = "D") -> np.ndarray:
    """Forecast ``horizon`` periods after the last history date at ``frequency``."""

    future_dates = pd.date_range(history["date"].max(), periods=horizon + 1, freq=frequency)[1:]
    return model.predict(pd.DataFrame({"ds": future_dates}))["yhat"].to_numpy()
