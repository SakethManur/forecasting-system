"""End-to-end training and prediction orchestration."""

from __future__ import annotations

from collections.abc import Callable
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.data_loader import load_sales_excel, prepare_daily_sales
from training.models.common import ModelResult
from training.reporting import generate_model_comparison_report
from training.visualization import plot_state_forecast
from utils.config import ensure_directories, settings
from utils.exceptions import DataValidationError, ForecastingError
from utils.logger import get_logger
from utils.model_io import (
    load_history,
    load_joblib_model,
    load_registry,
    save_history,
    save_json,
    save_registry,
    state_model_dir,
)


logger = get_logger(__name__)

SARIMA_NAME = "sarima"
PROPHET_NAME = "prophet"
XGBOOST_NAME = "xgboost"
LSTM_NAME = "lstm"
MODEL_NAMES = [SARIMA_NAME, PROPHET_NAME, XGBOOST_NAME, LSTM_NAME]


def train_all_states(
    excel_path: str | Path | None = None,
    model_names: list[str] | None = None,
) -> dict[str, Any]:
    """Train all requested model families and select the best model per state."""

    ensure_directories()
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("retail_sales_forecasting")

    selected_trainers = _selected_trainers(model_names)
    raw_df = load_sales_excel(excel_path)
    state_frames = prepare_daily_sales(raw_df)
    registry = {
        "states": {},
        "failed_states": {},
        "forecast_horizon_days": settings.forecast_horizon_days,
    }

    for state, state_df in state_frames.items():
        logger.info("Training models for state=%s rows=%s", state, len(state_df))
        try:
            _validate_state_history(state_df, state)
        except DataValidationError as exc:
            logger.exception("Skipping state=%s because validation failed", state)
            registry["failed_states"][state] = {"error": str(exc)}
            continue

        train_df, val_df = _time_series_split(state_df)
        save_history(state_df, state)

        state_results: dict[str, dict[str, Any]] = {}
        best_result: ModelResult | None = None

        for model_name, trainer in selected_trainers.items():
            try:
                logger.info("Starting model training state=%s model=%s", state, model_name)
                with mlflow.start_run(run_name=f"{state}-{model_name}", nested=False):
                    start_time = time.perf_counter()
                    result = trainer(train_df, val_df, state)
                    training_time = time.perf_counter() - start_time
                    _log_mlflow_run(
                        mlflow=mlflow,
                        result=result,
                        state=state,
                        train_rows=len(train_df),
                        validation_rows=len(val_df),
                        frequency=str(state_df.attrs.get("frequency", settings.frequency)),
                        step_days=int(state_df.attrs.get("step_days", 1)),
                        training_time=training_time,
                    )
                logger.info(
                    "Completed model training state=%s model=%s rmse=%.4f mae=%.4f training_time=%.2f",
                    state,
                    model_name,
                    result.metrics["rmse"],
                    result.metrics["mae"],
                    training_time,
                )
            except Exception as exc:
                logger.exception("Model failed for state=%s model=%s", state, model_name)
                state_results[model_name] = {"status": "failed", "error": str(exc)}
                continue

            state_results[model_name] = {
                "status": "trained",
                "metrics": result.metrics,
                "parameters": result.hyperparameters,
                "training_time": training_time,
                "artifact_path": str(result.artifact_path),
            }
            if best_result is None or result.metrics["rmse"] < best_result.metrics["rmse"]:
                best_result = result

        if best_result is None:
            raise ForecastingError(f"No model trained successfully for state: {state}")

        metadata = {
            "state": state,
            "best_model": best_result.name,
            "best_metrics": best_result.metrics,
            "models": state_results,
            "frequency": state_df.attrs.get("frequency", settings.frequency),
            "step_days": int(state_df.attrs.get("step_days", 1)),
            "last_train_date": str(train_df["date"].max().date()),
            "last_observed_date": str(state_df["date"].max().date()),
        }
        future_forecast_df = _predict_with_model(
            state=state,
            model_name=best_result.name,
            history=state_df,
            horizon_days=settings.forecast_horizon_days,
            metadata=metadata,
        )
        plot_path = plot_state_forecast(
            history_df=state_df,
            validation_df=val_df,
            validation_predictions=best_result.predictions,
            future_forecast_df=future_forecast_df,
            state=state,
            model_name=best_result.name,
            output_dir=settings.artifacts_dir / "plots",
        )
        metadata["plot_path"] = str(plot_path)
        logger.info("Saved forecast plot state=%s path=%s", state, plot_path)
        save_json(metadata, state_model_dir(state) / "metadata.json")
        registry["states"][state] = metadata
        logger.info(
            "Selected best model for state=%s model=%s rmse=%.4f",
            state,
            best_result.name,
            best_result.metrics["rmse"],
        )

    if not registry["states"]:
        raise ForecastingError("No states trained successfully.")

    report = generate_model_comparison_report(registry, settings.artifacts_dir)
    registry["model_comparison_report"] = report
    save_registry(registry)
    logger.info(
        "Saved model comparison report csv=%s json=%s",
        report["csv_path"],
        report["json_path"],
    )
    return registry


def predict_state(state: str, horizon_days: int | None = None) -> pd.DataFrame:
    """Forecast future sales for one state using its selected model."""

    requested_horizon_days = horizon_days or settings.forecast_horizon_days
    registry = load_registry()
    metadata = registry.get("states", {}).get(state)
    if not metadata:
        available = ", ".join(sorted(registry.get("states", {}).keys())) or "none"
        raise ForecastingError(f"Unknown state '{state}'. Available states: {available}")

    logger.info(
        "Prediction requested state=%s horizon_days=%s selected_model=%s",
        state,
        requested_horizon_days,
        metadata["best_model"],
    )
    history = load_history(state)
    return _predict_with_model(
        state=state,
        model_name=metadata["best_model"],
        history=history,
        horizon_days=requested_horizon_days,
        metadata=metadata,
    )


def _predict_with_model(
    state: str,
    model_name: str,
    history: pd.DataFrame,
    horizon_days: int,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    step_days = int(metadata.get("step_days", history.attrs.get("step_days", 1)))
    frequency = metadata.get("frequency", history.attrs.get("frequency", settings.frequency))
    history = history.copy()
    history.attrs["step_days"] = step_days
    history.attrs["frequency"] = frequency
    horizon = _horizon_periods(horizon_days, step_days)
    if model_name == SARIMA_NAME:
        from training.models.sarima_model import predict_sarima

        model = load_joblib_model(state, SARIMA_NAME)
        predictions = predict_sarima(model, horizon)
    elif model_name == PROPHET_NAME:
        from training.models.prophet_model import predict_prophet

        model = load_joblib_model(state, PROPHET_NAME)
        predictions = predict_prophet(model, horizon)
    elif model_name == XGBOOST_NAME:
        from training.models.xgboost_model import recursive_predict_xgboost

        model = load_joblib_model(state, XGBOOST_NAME)
        predictions = recursive_predict_xgboost(model, history, horizon)
    elif model_name == LSTM_NAME:
        from training.models.lstm_model import load_lstm_artifacts, recursive_predict_lstm

        model, scaler = load_lstm_artifacts(state)
        predictions = recursive_predict_lstm(model, scaler, history, horizon)
    else:
        raise ForecastingError(f"Unsupported registered model: {model_name}")

    future_dates = pd.date_range(
        history["date"].max() + pd.Timedelta(days=step_days),
        periods=horizon,
        freq=frequency,
    )
    return pd.DataFrame(
        {
            "date": future_dates,
            "state": state,
            "predicted_sales": np.maximum(predictions, 0.0),
            "model": model_name,
        }
    )


def predict_all_states(horizon_days: int | None = None) -> pd.DataFrame:
    registry = load_registry()
    frames = [
        predict_state(state, horizon_days)
        for state in sorted(registry.get("states", {}).keys())
    ]
    if not frames:
        raise ForecastingError("No trained models found. Run /train first.")
    return pd.concat(frames, ignore_index=True)


def list_models() -> dict[str, Any]:
    return load_registry()


def _selected_trainers(
    model_names: list[str] | None,
) -> dict[str, Callable[[pd.DataFrame, pd.DataFrame, str], ModelResult]]:
    available_trainers = _trainer_registry()
    if not model_names:
        return available_trainers
    unknown = sorted(set(model_names) - set(available_trainers))
    if unknown:
        raise ForecastingError(f"Unknown model name(s): {', '.join(unknown)}")
    return {name: available_trainers[name] for name in model_names}


def _trainer_registry() -> dict[str, Callable[[pd.DataFrame, pd.DataFrame, str], ModelResult]]:
    from training.models.lstm_model import train_lstm
    from training.models.prophet_model import train_prophet
    from training.models.sarima_model import train_sarima
    from training.models.xgboost_model import train_xgboost

    return {
        SARIMA_NAME: train_sarima,
        PROPHET_NAME: train_prophet,
        XGBOOST_NAME: train_xgboost,
        LSTM_NAME: train_lstm,
    }


def _validate_state_history(state_df: pd.DataFrame, state: str) -> None:
    step_days = int(state_df.attrs.get("step_days", 1))
    required_rows = _horizon_periods(settings.minimum_training_days, step_days) + _horizon_periods(
        settings.validation_days, step_days
    )
    if len(state_df) < required_rows:
        raise DataValidationError(
            f"State '{state}' has {len(state_df)} daily rows; at least "
            f"{required_rows} are required."
        )


def _time_series_split(state_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    step_days = int(state_df.attrs.get("step_days", 1))
    validation_periods = _horizon_periods(settings.validation_days, step_days)
    split_index = len(state_df) - validation_periods
    train_df = state_df.iloc[:split_index].reset_index(drop=True)
    val_df = state_df.iloc[split_index:].reset_index(drop=True)
    train_df.attrs.update(state_df.attrs)
    val_df.attrs.update(state_df.attrs)
    return train_df, val_df


def _horizon_periods(horizon_days: int, step_days: int) -> int:
    return max(1, int(np.ceil(horizon_days / max(step_days, 1))))


def _log_mlflow_run(
    mlflow,
    result: ModelResult,
    state: str,
    train_rows: int,
    validation_rows: int,
    frequency: str,
    step_days: int,
    training_time: float,
) -> None:
    """Log a complete model training run to MLflow."""

    mlflow.set_tag("pipeline", "retail_sales_forecasting")
    mlflow.set_tag("model_type", result.name)
    mlflow.set_tag("state", state)
    mlflow.log_param("model_type", result.name)
    mlflow.log_param("state", state)
    mlflow.log_param("train_rows", train_rows)
    mlflow.log_param("validation_rows", validation_rows)
    mlflow.log_param("frequency", frequency)
    mlflow.log_param("step_days", step_days)
    mlflow.log_params(_stringify_params(result.hyperparameters))
    mlflow.log_metric("RMSE", result.metrics["rmse"])
    mlflow.log_metric("MAE", result.metrics["mae"])
    mlflow.log_metric("training_time", training_time)
    if result.artifact_path.is_dir():
        mlflow.log_artifacts(str(result.artifact_path), artifact_path="model")
    else:
        mlflow.log_artifact(str(result.artifact_path), artifact_path="model")

    run_payload = {
        "state": state,
        "model_type": result.name,
        "metrics": {
            "RMSE": result.metrics["rmse"],
            "MAE": result.metrics["mae"],
            "training_time": training_time,
        },
        "parameters": result.hyperparameters,
        "artifact_path": str(result.artifact_path),
    }
    run_dir = result.artifact_path.parent if result.artifact_path.is_file() else result.artifact_path
    metrics_path = run_dir / f"{result.name}_mlflow_metrics.json"
    params_path = run_dir / f"{result.name}_mlflow_parameters.json"
    save_json(run_payload["metrics"], metrics_path)
    save_json(run_payload["parameters"], params_path)
    mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
    mlflow.log_artifact(str(params_path), artifact_path="parameters")


def _stringify_params(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value if isinstance(value, (str, int, float, bool)) else str(value)
        for key, value in parameters.items()
    }
