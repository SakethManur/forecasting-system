"""FastAPI request and response schemas.

These schemas drive request validation, JSON response shape, and Swagger docs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    excel_path: str | None = Field(
        default=None,
        description="Optional path to the Excel sales file. Defaults to data/sales.xlsx.",
        examples=["data/sales.xlsx"],
    )


class MetricValues(BaseModel):
    rmse: float = Field(description="Root mean squared error on validation data.")
    mae: float = Field(description="Mean absolute error on validation data.")


class ModelTrainingResult(BaseModel):
    status: str = Field(description="Training status for this model.")
    metrics: MetricValues | None = Field(
        default=None, description="Validation metrics when the model trained successfully."
    )
    parameters: dict[str, Any] | None = Field(
        default=None, description="Model hyperparameters logged to MLflow."
    )
    training_time: float | None = Field(
        default=None, description="Training duration in seconds."
    )
    artifact_path: str | None = Field(
        default=None, description="Saved model artifact path."
    )
    error: str | None = Field(default=None, description="Failure reason, if any.")


class StateTrainingMetrics(BaseModel):
    state: str
    best_model: str
    best_metrics: MetricValues
    models: dict[str, ModelTrainingResult]


class RankedModelSummary(BaseModel):
    rank: int
    model_name: str
    states_evaluated: int
    best_model_wins: int
    average_RMSE: float | None = None
    average_MAE: float | None = None
    average_training_time: float | None = None


class ModelComparisonReport(BaseModel):
    csv_path: str
    json_path: str
    ranked_summary: list[RankedModelSummary]


class TrainResponse(BaseModel):
    status: str = Field(examples=["completed"])
    metrics: list[StateTrainingMetrics]
    model_comparison_report: ModelComparisonReport | None = None
    failed_states: dict[str, Any] = Field(default_factory=dict)


class PredictRequest(BaseModel):
    state: str = Field(
        description="State to forecast. Must match one of the trained states.",
        examples=["Maharashtra"],
    )
    weeks_ahead: int = Field(
        default=8,
        ge=1,
        le=52,
        description="Number of future weeks to forecast.",
        examples=[8],
    )


class ForecastValue(BaseModel):
    week_start: str = Field(description="Forecast week start date in YYYY-MM-DD format.")
    state: str
    predicted_sales: float = Field(description="Forecast sales for the week.")


class PredictResponse(BaseModel):
    state: str
    weeks_ahead: int
    selected_best_model: str
    forecast_values: list[ForecastValue]


class TrainedModelSummary(BaseModel):
    state: str
    best_model: str
    best_metrics: MetricValues
    available_models: dict[str, ModelTrainingResult]


class ModelsResponse(BaseModel):
    models: list[TrainedModelSummary]
    failed_states: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    detail: Any
