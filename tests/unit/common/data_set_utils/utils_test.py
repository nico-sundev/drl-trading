from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ai_trading.common.model.asset_price_dataset import AssetPriceDataSet
from ai_trading.common.model.computed_dataset_container import ComputedDataSetContainer
from ai_trading.preprocess.data_set_utils.util import (
    detect_timeframe,
    ensure_datetime_time_column,
    separate_computed_datasets,
)


def test_separate_base_and_other_datasets():
    # Mock datasets
    base_dataset = MagicMock(spec=AssetPriceDataSet)
    base_dataset.base_dataset = True

    other_dataset_1 = MagicMock(spec=AssetPriceDataSet)
    other_dataset_1.base_dataset = False

    other_dataset_2 = MagicMock(spec=AssetPriceDataSet)
    other_dataset_2.base_dataset = False

    base_dataset_container = ComputedDataSetContainer(
        source_dataset=base_dataset, computed_dataframe=MagicMock()
    )
    other_dataset_container_1 = ComputedDataSetContainer(
        source_dataset=other_dataset_1, computed_dataframe=MagicMock()
    )
    other_dataset_container_2 = ComputedDataSetContainer(
        source_dataset=other_dataset_2, computed_dataframe=MagicMock()
    )

    datasets = [
        base_dataset_container,
        other_dataset_container_1,
        other_dataset_container_2,
    ]

    # Call the function
    result_base, result_others = separate_computed_datasets(datasets)

    # Assertions
    assert result_base == base_dataset_container
    assert other_dataset_container_1 in result_others
    assert other_dataset_container_2 in result_others
    assert len(result_others) == 2


def test_timeframe_detection() -> None:
    # Given
    df: pd.DataFrame = pd.DataFrame(
        {"Time": pd.date_range("2024-03-19 00:00:00", periods=4, freq="1h")}
    )

    # When
    detected_tf: pd.Timedelta = detect_timeframe(df)

    # Then
    assert detected_tf == pd.Timedelta("1h")


def test_timeframe_detection_empty_dataframe():
    """Test detect_timeframe with an empty DataFrame."""
    # Given
    empty_df = pd.DataFrame(columns=["Time"])

    # When/Then
    with pytest.raises(ValueError, match="DataFrame has less than 2 rows"):
        detect_timeframe(empty_df)


def test_timeframe_detection_none_dataframe():
    """Test detect_timeframe with None instead of a DataFrame."""
    # Given
    none_df = None

    # When/Then
    with pytest.raises(ValueError, match="DataFrame is None"):
        detect_timeframe(none_df)


def test_timeframe_detection_no_time_column():
    """Test detect_timeframe with a DataFrame missing the Time column."""
    # Given
    df_no_time = pd.DataFrame({"Value": [1, 2, 3]})

    # When/Then
    with pytest.raises(ValueError, match="'Time' column not found"):
        detect_timeframe(df_no_time)


def test_timeframe_detection_single_row():
    """Test detect_timeframe with a DataFrame containing only one row."""
    # Given
    single_row_df = pd.DataFrame({"Time": [pd.Timestamp("2023-01-01")]})

    # When/Then
    with pytest.raises(ValueError, match="less than 2 rows"):
        detect_timeframe(single_row_df)


# Tests for ensure_datetime_time_column
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
    with pytest.raises(ValueError, match="Could not convert 'Time' column to datetime"):
        ensure_datetime_time_column(df, "invalid time format dataframe")
