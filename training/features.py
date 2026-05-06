"""Feature engineering for supervised forecasting models."""

from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "lag_30",
    "rolling_mean_7",
    "rolling_std_7",
    "day_of_week",
    "month",
    "holiday_flag",
]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag, rolling, and calendar features from a sales series."""

    result = df.sort_values("date").copy()
    result["lag_1"] = result["sales"].shift(1)
    result["lag_7"] = result["sales"].shift(7)
    result["lag_30"] = result["sales"].shift(30)
    result["rolling_mean_7"] = result["sales"].shift(1).rolling(7).mean()
    result["rolling_std_7"] = result["sales"].shift(1).rolling(7).std()
    result["day_of_week"] = result["date"].dt.dayofweek
    result["month"] = result["date"].dt.month
    if "holiday_flag" in result.columns:
        result["holiday_flag"] = (
            pd.to_numeric(result["holiday_flag"], errors="coerce").fillna(0).astype(int)
        )
    else:
        result["holiday_flag"] = _holiday_flags(result["date"])
    return result


def make_supervised_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build a model-ready supervised frame and drop rows without lag history."""

    featured = add_time_features(df)
    return featured.dropna(subset=FEATURE_COLUMNS + ["sales"]).reset_index(drop=True)


def build_future_features(history: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Create recursive future features using observed/predicted sales history."""

    work = _history_columns(history)
    state = str(work["state"].iloc[-1])
    step_days = int(history.attrs.get("step_days", 1))
    future_rows = []

    for _ in range(horizon_days):
        next_date = work["date"].max() + pd.Timedelta(days=step_days)
        featured = add_time_features(work)
        row = featured.iloc[-1:].copy()
        future_row = row[FEATURE_COLUMNS].iloc[0].to_dict()
        future_row.update({"date": next_date, "state": state})
        future_rows.append(future_row)

        # Placeholder gets replaced by the caller with the model prediction.
        work = pd.concat(
            [
                work,
                pd.DataFrame(
                    [
                        {
                            "date": next_date,
                            "state": state,
                            "sales": work["sales"].iloc[-1],
                            "holiday_flag": 0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    return pd.DataFrame(future_rows)


def make_next_feature_row(history: pd.DataFrame, next_date: pd.Timestamp) -> pd.DataFrame:
    """Create a single feature row for recursive one-step forecasting."""

    temp = pd.concat(
        [
            _history_columns(history),
            pd.DataFrame(
                [
                    {
                        "date": next_date,
                        "state": history["state"].iloc[-1],
                        "sales": history["sales"].iloc[-1],
                        "holiday_flag": 0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    featured = add_time_features(temp)
    return featured.iloc[-1:][["date", "state"] + FEATURE_COLUMNS]


def _holiday_flags(dates: pd.Series) -> pd.Series:
    if dates.empty:
        return pd.Series(dtype=int)

    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(start=dates.min(), end=dates.max())
    return dates.dt.normalize().isin(holidays).astype(int)


def _history_columns(history: pd.DataFrame) -> pd.DataFrame:
    work = history.copy()
    if "holiday_flag" not in work.columns:
        work["holiday_flag"] = 0
    return work[["date", "state", "sales", "holiday_flag"]].sort_values("date").copy()
