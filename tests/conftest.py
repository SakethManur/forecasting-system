"""Shared test setup.

utils.config reads its settings from the environment once, at import time, so
every output directory is pointed at a temporary folder here, before any
project module is imported. Tests never touch the real data/, models/,
artifacts/, logs/ or mlruns/ folders.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="forecasting-tests-"))
for _name in ["DATA_DIR", "MODEL_DIR", "ARTIFACTS_DIR", "LOGS_DIR"]:
    os.environ[_name] = str(_TMP / _name.lower())
os.environ["MLFLOW_TRACKING_URI"] = (_TMP / "mlruns").as_uri()
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def make_weekly_sales(states=("Alpha", "Beta"), weeks: int = 160, seed: int = 0) -> pd.DataFrame:
    """Weekly sales rows shaped like the real dataset.

    Most rows fall on Sundays, some on the Saturday before (same week), and
    every fifth week is missing so the loader has gaps to interpolate.
    """

    rng = np.random.default_rng(seed)
    sundays = pd.date_range("2021-01-03", periods=weeks, freq="W-SUN")
    rows = []
    for index, state in enumerate(states):
        base = 1_000_000 * (index + 1)
        for week, sunday in enumerate(sundays):
            if week % 5 == 3:
                continue
            sales = base * (1 + 0.1 * np.sin(2 * np.pi * week / 52)) * rng.normal(1, 0.03)
            date = sunday - pd.Timedelta(days=1) if week % 7 == 0 else sunday
            rows.append(
                {
                    "date": date,
                    "state": state,
                    "sales": round(float(sales), 2),
                    "holiday_flag": int(week % 26 == 0),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def trained_registry(tmp_path_factory):
    """Train all four model families once on a small synthetic dataset."""

    from training.pipeline import train_all_states

    excel_path = tmp_path_factory.mktemp("data") / "sales.xlsx"
    make_weekly_sales().to_excel(excel_path, index=False)
    return train_all_states(excel_path)
