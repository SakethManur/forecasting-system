"""SARIMA training and forecasting."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from training.metrics import regression_metrics
from training.models.common import ModelResult
from utils.model_io import save_joblib_model


MODEL_NAME = "sarima"


def train_sarima(train_df: pd.DataFrame, val_df: pd.DataFrame, state: str) -> ModelResult:
    hyperparameters = {
        "order": "(1, 1, 1)",
        "seasonal_order": "(1, 1, 1, 7)",
        "enforce_stationarity": False,
        "enforce_invertibility": False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train_df["sales"],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)

    predictions = np.asarray(fitted.forecast(steps=len(val_df)))
    metrics = regression_metrics(val_df["sales"], predictions)
    artifact_path = save_joblib_model(fitted, state, MODEL_NAME)
    return ModelResult(MODEL_NAME, predictions, metrics, artifact_path, hyperparameters)


def predict_sarima(model, horizon_days: int) -> np.ndarray:
    return np.asarray(model.forecast(steps=horizon_days))
