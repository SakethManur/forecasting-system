"""Prophet training and forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from training.metrics import regression_metrics
from training.models.common import ModelResult
from utils.model_io import save_joblib_model


MODEL_NAME = "prophet"


def train_prophet(train_df: pd.DataFrame, val_df: pd.DataFrame, state: str) -> ModelResult:
    from prophet import Prophet

    prophet_train = train_df.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]]
    hyperparameters = {
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "interval_width": 0.95,
        "forecast_frequency": "D",
    }
    model = Prophet(
        yearly_seasonality=hyperparameters["yearly_seasonality"],
        weekly_seasonality=hyperparameters["weekly_seasonality"],
        daily_seasonality=hyperparameters["daily_seasonality"],
        interval_width=hyperparameters["interval_width"],
    )
    model.fit(prophet_train)

    future = model.make_future_dataframe(periods=len(val_df), freq="D")
    forecast = model.predict(future).tail(len(val_df))
    predictions = forecast["yhat"].to_numpy()
    metrics = regression_metrics(val_df["sales"], predictions)
    artifact_path = save_joblib_model(model, state, MODEL_NAME)
    return ModelResult(MODEL_NAME, predictions, metrics, artifact_path, hyperparameters)


def predict_prophet(model, horizon_days: int) -> np.ndarray:
    future = model.make_future_dataframe(periods=horizon_days, freq="D")
    return model.predict(future).tail(horizon_days)["yhat"].to_numpy()
