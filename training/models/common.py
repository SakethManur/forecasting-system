"""Shared model result types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ModelResult:
    name: str
    predictions: np.ndarray
    metrics: dict[str, float]
    artifact_path: Path
    hyperparameters: dict[str, Any]
