

import numpy as np
from drl_trading_common.utils.utils import ensure_datetime_time_column
import pandas as pd
import pytest
import warnings

from pandas.testing import assert_frame_equal

def test_ensure_datetime_time_column_with_existing_time_column():
    # Given
    # DataFrame with proper 'Time' column
    df = pd.DataFrame(
        {"Time": pd.date_range("2023-01-01", periods=5), "Value": np.random.randn(5)}
    )
    original_df = df.copy()

    # When
    result_df = ensure_datetime_time_column(df, "test dataframe")

    # Then
    assert "Time" in result_df.columns
    assert pd.api.types.is_datetime64_any_dtype(result_df["Time"])
    # Original df shouldn't be modified
    assert id(df) != id(result_df)
    assert_frame_equal(df, original_df)


def test_ensure_datetime_time_column_with_string_time_column():
    # Given
    # DataFrame with 'Time' as string
    df = pd.DataFrame(
        {"Time": ["2023-01-01", "2023-01-02", "2023-01-03"], "Value": [1.0, 2.0, 3.0]}
    )

    # When
    result_df = ensure_datetime_time_column(df, "string time dataframe")

    # Then
    assert "Time" in result_df.columns
    assert pd.api.types.is_datetime64_any_dtype(result_df["Time"])
    assert result_df["Time"][0].year == 2023
    assert result_df["Time"][0].month == 1
    assert result_df["Time"][0].day == 1


def test_ensure_datetime_time_column_with_datetime_index():
    # Given
    # DataFrame with DatetimeIndex but no 'Time' column
    dates = pd.date_range("2023-01-01", periods=5)
    df = pd.DataFrame({"Value": np.random.randn(5)}, index=dates)

    # When
    result_df = ensure_datetime_time_column(df, "index datetime dataframe")

    # Then
    assert "Time" in result_df.columns
    assert pd.api.types.is_datetime64_any_dtype(result_df["Time"])
    # Use pandas.testing.assert_series_equal instead of assert_frame_equal for Series objects
    pd.testing.assert_series_equal(
        result_df["Time"].reset_index(drop=True),
        pd.Series(dates, name="Time").reset_index(drop=True),
    )


def test_ensure_datetime_time_column_with_no_time_no_datetime_index():
    # Given
    # DataFrame with neither 'Time' column nor DatetimeIndex
    df = pd.DataFrame({"Value": np.random.randn(5)})

    # When/Then
    with pytest.raises(
        ValueError, match="must have a 'Time' column or a DatetimeIndex"
    ):
        ensure_datetime_time_column(df, "invalid dataframe")


def test_ensure_datetime_time_column_with_invalid_time_format():
    # Given
    # DataFrame with 'Time' column that can't be converted to datetime
    df = pd.DataFrame(
        {"Time": ["not-a-date", "invalid", "format"], "Value": [1.0, 2.0, 3.0]}
    )

    # When/Then
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Could not infer format")
        with pytest.raises(ValueError, match="Could not convert 'Time' column to datetime"):
            ensure_datetime_time_column(df, "invalid time format dataframe")
