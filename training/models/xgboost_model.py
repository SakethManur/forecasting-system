"""XGBoost training and recursive forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from training.features import FEATURE_COLUMNS, make_next_feature_row, make_supervised_frame
from training.metrics import regression_metrics
from training.models.common import ModelResult
from utils.config import settings
from utils.model_io import save_joblib_model


MODEL_NAME = "xgboost"


def train_xgboost(train_df: pd.DataFrame, val_df: pd.DataFrame, state: str) -> ModelResult:
    supervised = make_supervised_frame(train_df)
    hyperparameters = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "reg:squarederror",
        "random_state": settings.random_state,
        "features": ",".join(FEATURE_COLUMNS),
    }
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                XGBRegressor(
                    n_estimators=hyperparameters["n_estimators"],
                    max_depth=hyperparameters["max_depth"],
                    learning_rate=hyperparameters["learning_rate"],
                    subsample=hyperparameters["subsample"],
                    colsample_bytree=hyperparameters["colsample_bytree"],
                    objective=hyperparameters["objective"],
                    random_state=hyperparameters["random_state"],
                ),
            ),
        ]
    )
    model.fit(supervised[FEATURE_COLUMNS], supervised["sales"])
    predictions = recursive_predict_xgboost(model, train_df, len(val_df))
    metrics = regression_metrics(val_df["sales"], predictions)
    artifact_path = save_joblib_model(model, state, MODEL_NAME)
    return ModelResult(MODEL_NAME, predictions, metrics, artifact_path, hyperparameters)


def recursive_predict_xgboost(model, history: pd.DataFrame, horizon_days: int) -> np.ndarray:
    work = history.copy()
    if "holiday_flag" not in work.columns:
        work["holiday_flag"] = 0
    work = work[["date", "state", "sales", "holiday_flag"]].sort_values("date").copy()
    step_days = int(history.attrs.get("step_days", 1))
    predictions = []

    for _ in range(horizon_days):
        next_date = work["date"].max() + pd.Timedelta(days=step_days)
        feature_row = make_next_feature_row(work, next_date)
        prediction = float(model.predict(feature_row[FEATURE_COLUMNS])[0])
        prediction = max(0.0, prediction)
        predictions.append(prediction)
        work = pd.concat(
            [
                work,
                pd.DataFrame(
                    [
                        {
                            "date": next_date,
                            "state": work["state"].iloc[-1],
                            "sales": prediction,
                            "holiday_flag": 0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    return np.asarray(predictions)
