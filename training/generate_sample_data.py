"""Generate a realistic sample retail sales dataset for Indian states."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import settings
from utils.logger import get_logger


logger = get_logger(__name__)

INDIAN_STATES = [
    "Maharashtra",
    "Karnataka",
    "Tamil Nadu",
    "Gujarat",
    "Delhi",
    "West Bengal",
    "Rajasthan",
    "Uttar Pradesh",
]

STATE_BASE_SALES = {
    "Maharashtra": 185000,
    "Karnataka": 150000,
    "Tamil Nadu": 142000,
    "Gujarat": 134000,
    "Delhi": 125000,
    "West Bengal": 118000,
    "Rajasthan": 104000,
    "Uttar Pradesh": 160000,
}

FESTIVAL_WEEKS = {
    "2022-01-10",
    "2022-03-14",
    "2022-08-08",
    "2022-10-17",
    "2022-10-24",
    "2022-12-26",
    "2023-01-09",
    "2023-03-06",
    "2023-08-14",
    "2023-11-06",
    "2023-11-13",
    "2023-12-25",
    "2024-01-15",
    "2024-03-25",
    "2024-08-12",
    "2024-10-28",
    "2024-11-04",
    "2024-12-23",
}


def generate_dataset(output_path: str | Path | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-03", "2024-12-30", freq="W-MON")
    rows = []

    for state_index, state in enumerate(INDIAN_STATES):
        base_sales = STATE_BASE_SALES[state]
        state_growth = 0.0018 + (state_index * 0.00012)

        for week_index, date in enumerate(dates):
            yearly_seasonality = 1 + 0.12 * np.sin(2 * np.pi * week_index / 52)
            quarterly_cycle = 1 + 0.05 * np.cos(2 * np.pi * week_index / 13)
            trend = 1 + state_growth * week_index
            holiday_flag = int(date.strftime("%Y-%m-%d") in FESTIVAL_WEEKS)
            holiday_boost = 1.28 if holiday_flag else 1.0
            monsoon_softness = 0.93 if date.month in {6, 7} else 1.0
            random_noise = rng.normal(1.0, 0.075)

            sales = (
                base_sales
                * trend
                * yearly_seasonality
                * quarterly_cycle
                * holiday_boost
                * monsoon_softness
                * random_noise
            )

            rows.append(
                {
                    "date": date,
                    "state": state,
                    "sales": round(max(sales, 0), 2),
                    "holiday_flag": holiday_flag,
                }
            )

    df = pd.DataFrame(rows)
    df = _inject_missing_dates_and_values(df, rng)

    path = Path(output_path) if output_path else settings.data_dir / "sales.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)
    logger.info("Generated sample sales dataset: %s rows=%s", path, len(df))
    return df


def _inject_missing_dates_and_values(
    df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Drop some state/week rows and null some sales values intentionally."""

    result = df.copy()
    missing_date_indices = []
    missing_value_indices = []

    for state in INDIAN_STATES:
        state_indices = result.index[result["state"] == state].to_numpy()
        missing_date_indices.extend(
            rng.choice(state_indices, size=5, replace=False).tolist()
        )
        remaining_indices = np.setdiff1d(state_indices, missing_date_indices)
        missing_value_indices.extend(
            rng.choice(remaining_indices, size=4, replace=False).tolist()
        )

    result = result.drop(index=missing_date_indices).reset_index(drop=True)
    adjusted_missing_value_indices = rng.choice(
        result.index.to_numpy(), size=len(missing_value_indices), replace=False
    )
    result.loc[adjusted_missing_value_indices, "sales"] = np.nan
    return result.sort_values(["state", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    generate_dataset()
