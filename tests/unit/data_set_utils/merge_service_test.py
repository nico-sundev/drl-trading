from datetime import datetime

import pandas as pd
import pytest

from ai_trading.data_set_utils.merge_service import MergeService, MergeServiceInterface


@pytest.fixture
def merge_service() -> MergeServiceInterface:
    """Fixture providing a MergeService instance."""
    return MergeService()


@pytest.fixture
def base_timeframe_data() -> pd.DataFrame:
    """Fixture providing H1 (1-hour) timeframe data."""
    # Create a sample dataset with hourly candles
    times = [
        datetime(2023, 1, 1, 10, 0),
        datetime(2023, 1, 1, 11, 0),
        datetime(2023, 1, 1, 12, 0),
        datetime(2023, 1, 1, 13, 0),
        datetime(2023, 1, 1, 14, 0),
        datetime(2023, 1, 1, 15, 0),
        datetime(2023, 1, 1, 16, 0),
    ]

    data = {
        "Time": times,
        "Open": [100, 105, 110, 115, 120, 125, 130],
        "High": [105, 110, 115, 120, 125, 130, 135],
        "Low": [95, 100, 105, 110, 115, 120, 125],
        "Close": [105, 110, 115, 120, 125, 130, 135],
        "Volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600],
    }

    return pd.DataFrame(data)


@pytest.fixture
def higher_timeframe_data() -> pd.DataFrame:
    """Fixture providing H4 (4-hour) timeframe data with some technical indicators."""
    # Create a sample dataset with 4-hour candles
    times = [
        datetime(2023, 1, 1, 8, 0),  # This candle closes at 12:00
        datetime(2023, 1, 1, 12, 0),  # This candle closes at 16:00
    ]

    data = {
        "Time": times,
        "Open": [90, 110],
        "High": [110, 130],
        "Low": [85, 105],
        "Close": [110, 130],
        "Volume": [5000, 6000],
        "RSI": [55, 65],
        "MA20": [105, 115],
        "Bollinger_Upper": [115, 125],
        "Bollinger_Lower": [95, 105],
    }

    return pd.DataFrame(data)


def test_merge_timeframes_basic(
    merge_service: MergeServiceInterface,
    base_timeframe_data: pd.DataFrame,
    higher_timeframe_data: pd.DataFrame,
):
    """Test the basic merging functionality."""
    # Given
    base_df = base_timeframe_data
    higher_df = higher_timeframe_data
    expected_row_count = len(base_df)

    # When
    result_df = merge_service.merge_timeframes(base_df, higher_df)

    # Then
    # Check that row count is preserved from base dataframe
    assert (
        len(result_df) == expected_row_count
    ), "Merged dataframe should have same number of rows as base dataframe"

    # Verify the Time column is preserved
    assert "Time" in result_df.columns, "Time column should be preserved"
    assert list(result_df["Time"]) == list(
        base_df["Time"]
    ), "Time values should be preserved"

    # Check that higher timeframe columns are properly prefixed
    assert (
        "HTF-240_RSI" in result_df.columns
    ), "Higher timeframe columns should have proper prefix"
    assert (
        "HTF-240_MA20" in result_df.columns
    ), "Higher timeframe columns should have proper prefix"

    # OHLCV columns from higher timeframe should not be included
    assert (
        "HTF-240_Open" not in result_df.columns
    ), "OHLCV columns from higher timeframe should not be included"
    assert (
        "HTF-240_Close" not in result_df.columns
    ), "OHLCV columns from higher timeframe should not be included"


def test_merge_timeframes_data_alignment(
    merge_service: MergeServiceInterface,
    base_timeframe_data: pd.DataFrame,
    higher_timeframe_data: pd.DataFrame,
):
    """Test that data is properly aligned with respect to time to prevent lookahead bias."""
    # Given
    base_df = base_timeframe_data
    higher_df = higher_timeframe_data

    # When
    result_df = merge_service.merge_timeframes(base_df, higher_df)

    # Then
    # For timestamps before the first higher timeframe candle closes, values should be NaN
    # First H4 candle closes at 12:00, so rows before that should have NaN
    assert pd.isna(
        result_df.loc[0, "HTF-240_RSI"]
    ), "Data before first higher timeframe candle closes should be NaN"
    assert pd.isna(
        result_df.loc[1, "HTF-240_RSI"]
    ), "Data before first higher timeframe candle closes should be NaN"

    # At 12:00 and later until 16:00, should have first higher timeframe candle data
    assert (
        result_df.loc[2, "HTF-240_RSI"] == 55
    ), "Should use first higher timeframe candle data at 12:00"
    assert (
        result_df.loc[3, "HTF-240_RSI"] == 55
    ), "Should use first higher timeframe candle data at 13:00"
    assert (
        result_df.loc[4, "HTF-240_RSI"] == 55
    ), "Should use first higher timeframe candle data at 14:00"
    assert (
        result_df.loc[5, "HTF-240_RSI"] == 55
    ), "Should use first higher timeframe candle data at 15:00"

    # At 16:00 and later, should have second higher timeframe candle data
    assert (
        result_df.loc[6, "HTF-240_RSI"] == 65
    ), "Should use second higher timeframe candle data at 16:00"


def test_merge_timeframes_unsorted_data(
    merge_service: MergeServiceInterface,
    base_timeframe_data: pd.DataFrame,
    higher_timeframe_data: pd.DataFrame,
):
    """Test that the merge works correctly with unsorted input data."""
    # Given
    # Create unsorted versions of the dataframes
    unsorted_base_df = base_timeframe_data.sample(frac=1).reset_index(drop=True)
    unsorted_higher_df = higher_timeframe_data.sample(frac=1).reset_index(drop=True)

    # When
    result_unsorted = merge_service.merge_timeframes(
        unsorted_base_df, unsorted_higher_df
    )
    result_sorted = merge_service.merge_timeframes(
        base_timeframe_data, higher_timeframe_data
    )

    # Then
    # Sort both results by Time to ensure we can compare them properly
    result_unsorted = result_unsorted.sort_values("Time").reset_index(drop=True)
    result_sorted = result_sorted.sort_values("Time").reset_index(drop=True)

    # The results should be identical regardless of the input sorting
    pd.testing.assert_frame_equal(
        result_unsorted,
        result_sorted,
        "Results should be identical regardless of input sorting",
    )


def test_merge_timeframes_empty_higher_data(
    merge_service: MergeServiceInterface, base_timeframe_data: pd.DataFrame
):
    """Test merging with empty higher timeframe data."""
    # Given
    base_df = base_timeframe_data
    empty_higher_df = pd.DataFrame(
        columns=["Time", "Open", "High", "Low", "Close", "Volume", "RSI"]
    )

    # When
    # This should now handle the ValueError gracefully inside MergeService
    result_df = merge_service.merge_timeframes(base_df, empty_higher_df)

    # Then
    # Should return a dataframe with just the Time column from the base dataframe
    assert len(result_df) == len(base_df), "Should preserve base DataFrame row count"
    assert "Time" in result_df.columns, "Should preserve Time column"
    assert list(result_df["Time"]) == list(
        base_df["Time"]
    ), "Time values should be preserved"

    # No higher timeframe columns should be added
    assert list(result_df.columns) == [
        "Time"
    ], "Should only have Time column in the result"
