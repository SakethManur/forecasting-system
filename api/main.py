"""FastAPI application for training and serving sales forecasts."""

from __future__ import annotations

import time

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas import (
    ErrorResponse,
    ModelsResponse,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)
from training.pipeline import list_models, predict_state, train_all_states
from utils.config import ensure_directories
from utils.exceptions import ForecastingError
from utils.logger import configure_logging, get_logger


configure_logging()
logger = get_logger(__name__)
app = FastAPI(
    title="Sales Forecasting System",
    description=(
        "Production API for state-level retail sales forecasting. "
        "The service trains SARIMA, Prophet, XGBoost, and LSTM models per state, "
        "selects the best model by validation RMSE, and serves weekly forecasts."
    ),
    version="1.0.0",
    contact={"name": "Forecasting API Support"},
)


@app.on_event("startup")
def startup() -> None:
    ensure_directories()
    logger.info("Application startup complete")


@app.middleware("http")
async def log_endpoint_activity(request: Request, call_next):
    start_time = time.perf_counter()
    client_host = request.client.host if request.client else "unknown"
    logger.info(
        "Endpoint request method=%s path=%s client=%s",
        request.method,
        request.url.path,
        client_host,
    )
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            "Endpoint error method=%s path=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Endpoint response method=%s path=%s status_code=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/health")
def health() -> dict[str, str]:
    logger.info("Health check requested")
    return {"status": "ok"}


@app.exception_handler(ForecastingError)
async def forecasting_error_handler(
    request: Request, exc: ForecastingError
) -> JSONResponse:
    logger.error("Forecasting error on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
    logger.error("Missing file on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    logger.warning("Invalid request on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.post(
    "/train",
    response_model=TrainResponse,
    summary="Train Forecasting Models",
    description=(
        "Trains SARIMA, Prophet, XGBoost, and LSTM models for every detected state "
        "in the Excel dataset. Returns validation RMSE/MAE and the selected best "
        "model for each state."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Dataset or training error."},
        404: {"model": ErrorResponse, "description": "Excel file not found."},
        500: {"model": ErrorResponse, "description": "Unexpected server error."},
    },
)
def train(request: TrainRequest) -> TrainResponse:
    logger.info("Received training request excel_path=%s", request.excel_path)
    registry = train_all_states(request.excel_path)
    metrics = _training_metrics(registry)
    logger.info("Training completed states=%s", len(metrics))
    return TrainResponse(
        status="completed",
        metrics=metrics,
        model_comparison_report=registry.get("model_comparison_report"),
        failed_states=registry.get("failed_states", {}),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict Weekly Sales",
    description=(
        "Forecasts sales for one trained state for the requested number of weeks. "
        "The response includes weekly forecast values and the selected best model."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Unknown state or missing model."},
        422: {"model": ErrorResponse, "description": "Invalid request payload."},
        500: {"model": ErrorResponse, "description": "Unexpected server error."},
    },
)
def predict(request: PredictRequest) -> PredictResponse:
    logger.info(
        "Received prediction request state=%s weeks_ahead=%s",
        request.state,
        request.weeks_ahead,
    )
    horizon_days = request.weeks_ahead * 7
    forecast_df = predict_state(request.state, horizon_days)
    selected_model = str(forecast_df["model"].iloc[0])
    return PredictResponse(
        state=request.state,
        weeks_ahead=request.weeks_ahead,
        selected_best_model=selected_model,
        forecast_values=_serialize_weekly(forecast_df),
    )


@app.get(
    "/models",
    response_model=ModelsResponse,
    summary="List Trained Models",
    description=(
        "Returns available trained models, validation metrics, and the best model "
        "selected for each state."
    ),
    responses={500: {"model": ErrorResponse, "description": "Unexpected server error."}},
)
def models() -> ModelsResponse:
    logger.info("Received models listing request")
    registry = list_models()
    return ModelsResponse(
        models=_model_summaries(registry),
        failed_states=registry.get("failed_states", {}),
    )


def _serialize_weekly(forecast_df: pd.DataFrame) -> list[dict]:
    weekly = forecast_df.copy()
    weekly["week_start"] = weekly["date"].dt.to_period("W-SUN").dt.start_time
    weekly = (
        weekly.groupby(["state", "model", "week_start"], as_index=False)[
            "predicted_sales"
        ]
        .sum()
        .sort_values(["state", "week_start"])
    )
    weekly["week_start"] = weekly["week_start"].dt.strftime("%Y-%m-%d")
    weekly["predicted_sales"] = weekly["predicted_sales"].round(4)
    return weekly[["week_start", "state", "predicted_sales"]].to_dict(orient="records")


def _training_metrics(registry: dict) -> list[dict]:
    return [
        {
            "state": state,
            "best_model": metadata["best_model"],
            "best_metrics": metadata["best_metrics"],
            "models": metadata["models"],
        }
        for state, metadata in sorted(registry.get("states", {}).items())
    ]


def _model_summaries(registry: dict) -> list[dict]:
    return [
        {
            "state": state,
            "best_model": metadata["best_model"],
            "best_metrics": metadata["best_metrics"],
            "available_models": metadata["models"],
        }
        for state, metadata in sorted(registry.get("states", {}).items())
    ]
