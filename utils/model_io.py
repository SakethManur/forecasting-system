"""Model artifact persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils.config import settings
from utils.exceptions import ModelNotFoundError


REGISTRY_FILE = settings.model_dir / "model_registry.json"


def safe_state_name(state: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in state.lower()).strip("_")


def state_model_dir(state: str) -> Path:
    return settings.model_dir / safe_state_name(state)


def save_joblib_model(model: Any, state: str, model_name: str) -> Path:
    import joblib

    path = state_model_dir(state)
    path.mkdir(parents=True, exist_ok=True)
    artifact_path = path / f"{model_name}.joblib"
    joblib.dump(model, artifact_path)
    return artifact_path


def load_joblib_model(state: str, model_name: str) -> Any:
    import joblib

    artifact_path = state_model_dir(state) / f"{model_name}.joblib"
    if not artifact_path.exists():
        raise ModelNotFoundError(f"Model artifact not found: {artifact_path}")
    return joblib.load(artifact_path)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ModelNotFoundError(f"Metadata not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_history(df: pd.DataFrame, state: str) -> Path:
    path = state_model_dir(state)
    path.mkdir(parents=True, exist_ok=True)
    history_path = path / "history.csv"
    df.to_csv(history_path, index=False)
    return history_path


def load_history(state: str) -> pd.DataFrame:
    history_path = state_model_dir(state) / "history.csv"
    if not history_path.exists():
        raise ModelNotFoundError(f"History artifact not found: {history_path}")
    history = pd.read_csv(history_path)
    history["date"] = pd.to_datetime(history["date"])
    return history


def save_registry(registry: dict[str, Any]) -> None:
    save_json(registry, REGISTRY_FILE)


def load_registry() -> dict[str, Any]:
    if not REGISTRY_FILE.exists():
        return {"states": {}}
    return load_json(REGISTRY_FILE)
