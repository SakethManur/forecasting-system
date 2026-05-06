"""Application configuration.

Environment variables can override all paths and dataset column names so the
same code can run locally, in CI, or in a container without source changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
    model_dir: Path = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
    artifacts_dir: Path = Path(os.getenv("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
    logs_dir: Path = Path(os.getenv("LOGS_DIR", PROJECT_ROOT / "logs"))
    mlflow_tracking_uri: str = os.getenv(
        "MLFLOW_TRACKING_URI", f"file:///{(PROJECT_ROOT / 'mlruns').as_posix()}"
    )
    mlflow_experiment_name: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "retail_sales_forecasting"
    )
    default_excel_path: Path = Path(
        os.getenv("SALES_EXCEL_PATH", PROJECT_ROOT / "data" / "sales.xlsx")
    )
    date_column: str = os.getenv("DATE_COLUMN", "date")
    state_column: str = os.getenv("STATE_COLUMN", "state")
    sales_column: str = os.getenv("SALES_COLUMN", "sales")
    frequency: str = os.getenv("FORECAST_FREQUENCY", "D")
    forecast_horizon_days: int = int(os.getenv("FORECAST_HORIZON_DAYS", "56"))
    validation_days: int = int(os.getenv("VALIDATION_DAYS", "56"))
    minimum_training_days: int = int(os.getenv("MINIMUM_TRAINING_DAYS", "90"))
    lstm_sequence_length: int = int(os.getenv("LSTM_SEQUENCE_LENGTH", "30"))
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))


settings = Settings()


def ensure_directories() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (settings.artifacts_dir / "plots").mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
