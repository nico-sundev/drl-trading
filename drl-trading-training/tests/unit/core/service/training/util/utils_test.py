from drl_trading_training.core.service.training.util.util import detect_timeframe
import pandas as pd
import pytest

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


def test_timeframe_detection_no_time_column_and_index():
    """Test detect_timeframe with a DataFrame missing the Time column."""
    # Given
    df_no_time = pd.DataFrame({"Value": [1, 2, 3]})

    # When/Then
    with pytest.raises(ValueError):
        detect_timeframe(df_no_time)


def test_timeframe_detection_single_row():
    """Test detect_timeframe with a DataFrame containing only one row."""
    # Given
    single_row_df = pd.DataFrame({"Time": [pd.Timestamp("2023-01-01")]})

    # When/Then
    with pytest.raises(ValueError, match="less than 2 rows"):
        detect_timeframe(single_row_df)
