"""Tests for dataset loading and weekly bucketing."""

from __future__ import annotations

import pandas as pd
import pytest

from training.data_loader import (
    detect_sales_columns,
    infer_frequency,
    load_sales_excel,
    prepare_daily_sales,
)
from utils.exceptions import DataValidationError


def _rows(records):
    df = pd.DataFrame(records, columns=["date", "state", "sales"])
    df["date"] = pd.to_datetime(df["date"])
    df["holiday_flag"] = 0
    return df


def test_weeks_are_anchored_on_most_common_weekday_not_first_row():
    # First row is a Saturday, but most rows are Sundays (the real dataset's shape).
    dates = pd.Series(pd.to_datetime(["2023-01-07"]))
    dates = pd.concat([dates, pd.Series(pd.date_range("2023-01-15", periods=10, freq="W-SUN"))])
    assert infer_frequency(dates) == ("W-SUN", 7)


@pytest.mark.parametrize(
    ("dates", "expected"),
    [
        (pd.date_range("2023-01-01", periods=30, freq="D"), ("D", 1)),
        (pd.date_range("2023-01-01", periods=12, freq="MS"), ("MS", 30)),
    ],
)
def test_infer_frequency_detects_daily_and_monthly(dates, expected):
    assert infer_frequency(pd.Series(dates)) == expected


def test_rows_off_the_anchor_day_are_kept_and_averaged():
    # Regression test: the loader used to drop every row that did not fall
    # exactly on the weekly grid's anchor day.
    df = _rows(
        [
            ("2023-01-01", "S", 100.0),
            ("2023-01-08", "S", 200.0),
            ("2023-01-14", "S", 300.0),  # Saturday, same week as the Sunday below
            ("2023-01-15", "S", 500.0),
            ("2023-01-22", "S", 400.0),
        ]
    )
    series = prepare_daily_sales(df)["S"].set_index("date")

    assert series.loc["2023-01-15", "sales"] == pytest.approx(400.0)  # mean of 300 and 500
    assert series["is_observed"].all()
    assert len(series) == 4


def test_missing_weeks_are_interpolated_and_marked_unobserved():
    df = _rows(
        [
            ("2023-01-01", "S", 100.0),
            ("2023-01-08", "S", 200.0),
            ("2023-01-22", "S", 400.0),  # 2023-01-15 is missing
            ("2023-01-29", "S", 500.0),
        ]
    )
    frame = prepare_daily_sales(df)["S"]
    gap = frame.set_index("date").loc["2023-01-15"]

    assert frame.attrs == {"frequency": "W-SUN", "step_days": 7}
    assert not gap["is_observed"]
    assert gap["sales"] == pytest.approx(300.0)
    assert int(frame["is_observed"].sum()) == 4


def test_duplicate_state_date_rows_are_summed():
    df = _rows(
        [
            ("2023-01-01", "S", 100.0),
            ("2023-01-08", "S", 150.0),
            ("2023-01-08", "S", 50.0),
            ("2023-01-15", "S", 300.0),
        ]
    )
    frame = prepare_daily_sales(df)["S"].set_index("date")
    assert frame.loc["2023-01-08", "sales"] == pytest.approx(200.0)


def test_each_state_gets_its_own_series():
    df = _rows(
        [("2023-01-01", "A", 1.0), ("2023-01-08", "A", 2.0), ("2023-01-01", "B", 3.0), ("2023-01-08", "B", 4.0)]
    )
    assert sorted(prepare_daily_sales(df)) == ["A", "B"]


def test_prepare_rejects_frames_without_required_columns():
    with pytest.raises(DataValidationError):
        prepare_daily_sales(pd.DataFrame({"date": [], "sales": []}))


def test_load_sales_excel_detects_common_column_names(tmp_path):
    path = tmp_path / "sales.xlsx"
    pd.DataFrame(
        {
            "Order Date": pd.date_range("2023-01-01", periods=3, freq="W-SUN"),
            "State": [" Texas ", "Texas", "Texas"],
            "Revenue": [10.0, 20.0, 30.0],
        }
    ).to_excel(path, index=False)

    df = load_sales_excel(path)

    assert list(df.columns) == ["date", "state", "sales", "holiday_flag"]
    assert df["state"].unique().tolist() == ["Texas"]
    assert df["holiday_flag"].eq(0).all()
    assert df["sales"].sum() == pytest.approx(60.0)


def test_load_sales_excel_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sales_excel(tmp_path / "nope.xlsx")


def test_detect_sales_columns_names_what_is_missing():
    with pytest.raises(DataValidationError, match="state"):
        detect_sales_columns(pd.DataFrame({"date": ["2023-01-01"], "sales": [1.0]}))
