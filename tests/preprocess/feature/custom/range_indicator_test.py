from unittest.mock import MagicMock
import pytest
import pandas as pd
import numpy as np

from ai_trading.preprocess.feature.custom.enum.wick_handle_strategy_enum import WICK_HANDLE_STRATEGY
from ai_trading.preprocess.feature.custom.range_indicator import PIVOT_HIGH, PIVOT_LOW, SupportResistanceFinder


@pytest.fixture
def mock_config():
    """Fixture to create a mock config object"""
    mock_config = MagicMock()
    mock_config.macd.fast = 3
    mock_config.macd.slow = 6
    mock_config.macd.signal = 5
    mock_config.roc_lengths = [14, 7, 3]
    mock_config.rsi_lengths = [14, 3]
    mock_config.range.lookback = 5
    mock_config.range.wick_handle_strategy = WICK_HANDLE_STRATEGY.LAST_WICK_ONLY
    return mock_config


@pytest.fixture
def mock_data():
    """Creates a small mock DataFrame for testing support and resistance."""
    return pd.DataFrame(
        {
            "Open": [99, 100, 102, 101, 105, 103, 108, 106, 110, 97.3, 99.4],
            "Close": [100, 102, 101, 105, 103, 108, 106, 110, 97.3, 99.4, 100.8],
            "High": [101, 103, 102, 107, 104, 110, 107, 113, 117.5, 101.2, 101],
            "Low": [99, 100, 100, 104, 102, 107, 104, 104, 95.2, 96.8, 98.3],
        }
    )

@pytest.fixture
def mock_pivot_cache():
    """Creates a small mock DataFrame for testing support and resistance."""
    return pd.DataFrame(
        {
            "index": [1, 2, 3],
            "top": [106, 117.5, 97.3],
            "bottom": [104, 110, 96.8],
            "type": [PIVOT_LOW, PIVOT_HIGH, PIVOT_LOW],
        }
    )


@pytest.fixture
def mock_finder(mock_data, mock_config):
    """Returns an instance of SupportResistanceFinder with a small lookback."""
    return SupportResistanceFinder(mock_data, mock_config)


def test_validate_dataframe_success(mock_finder):
    """Test that validation passes for a valid DataFrame."""
    mock_finder.validate_dataframe()


def test_validate_dataframe_missing_columns(mock_finder):
    """Test that validation fails when required columns are missing."""
    df = pd.DataFrame({"Close": [100, 101, 102]})  # Missing "High" and "Low"
    mock_finder.source_data_frame = df
    with pytest.raises(ValueError, match="Missing required columns"):
        mock_finder.validate_dataframe()


def test_validate_dataframe_empty(mock_finder):
    """Test that validation fails on an empty DataFrame."""
    df = pd.DataFrame(columns=["Close", "High", "Low"])
    mock_finder.source_data_frame = df
    with pytest.raises(ValueError, match="DataFrame is empty"):
        mock_finder.validate_dataframe()


def test_validate_dataframe_nan_values(mock_finder):
    """Test that validation fails when NaN values are present."""
    df = pd.DataFrame(
        {"Close": [100, np.nan, 102], "High": [101, 103, 102], "Low": [99, 100, 100]}
    )
    mock_finder.source_data_frame = df
    with pytest.raises(ValueError, match="DataFrame contains NaN values"):
        mock_finder.validate_dataframe()


def test_find_pivot_points(mock_finder, mock_data):
    """Test that pivot highs and lows are correctly identified."""
    expected_highs = [False, False, True, False, True, False, True, False, True, False, False]
    expected_lows = [False, False, False, True, False, True, False, True, False, True, False]
    
    for index, row in mock_data.iterrows():
        [found_pivot_high, found_pivot_low] = mock_finder.find_pivot_points(index)
        assert found_pivot_high == expected_highs[index]
        assert found_pivot_low == expected_lows[index]

def test_clean_pivot_cache(mock_finder, mock_data, mock_pivot_cache):
    last_close = mock_data.iloc[-1]["Close"]
    mock_finder.pivot_cache = mock_pivot_cache
    assert len(mock_finder.pivot_cache) == 3
    mock_finder.clean_pivot_cache(last_close)
    assert len(mock_finder.pivot_cache) == 2

def test_find_next_zone(mock_finder, mock_data, mock_pivot_cache):
    """Test that the correct support and resistance levels are found."""
    last_close = mock_data.iloc[-1]["Close"]
    mock_finder.pivot_cache = mock_pivot_cache
    resistance = mock_finder.find_next_zone(PIVOT_HIGH, last_close)
    support = mock_finder.find_next_zone(PIVOT_LOW, last_close)

    # Check if valid support & resistance zones are found
    assert not np.isnan(support["top"])
    assert not np.isnan(support["bottom"])
    assert not np.isnan(resistance["top"])
    assert not np.isnan(resistance["bottom"])
    assert support["top"] == 97.3
    assert resistance["bottom"] == 110


# def test_calculate_wick_threshold():
#     """Test that the wick threshold calculation is correct."""
#     close, extreme = 100, 110
#     assert (
#         SupportResistanceFinder.calculate_wick_threshold(close, extreme, ratio=0.5)
#         == 105
#     )

#     close = 100
#     extreme = 110
#     expected_value = 106.67
#     ratio = 2 / 3
#     tolerance_percentage = 0.02  # 2% tolerance

#     result = SupportResistanceFinder.calculate_wick_threshold(close, extreme, ratio)

#     # Calculate the acceptable range based on percentage tolerance
#     lower_bound = expected_value * (1 - tolerance_percentage)
#     upper_bound = expected_value * (1 + tolerance_percentage)

#     # Assert that the result is within the tolerance percentage range
#     assert (
#         lower_bound <= result <= upper_bound
#     ), f"Expected {expected_value} ± {tolerance_percentage*100}%, but got {result}"


def test_find_support_resistance_zones(mock_finder):
    """Test the full support & resistance zone detection process."""
    result = mock_finder.find_support_resistance_zones()

    assert "support_range" in result.columns
    assert "resistance_range" in result.columns

    last_row = result.iloc[-1]
    assert not np.isnan(last_row["support_range"]) or not np.isnan(
        last_row["resistance_range"]
    )


def test_no_zone_found_within_lookback(mock_finder):
    """Test that NaN is returned when no support/resistance is found in the lookback range."""
    df = pd.DataFrame(
        {
            "Open": [99, 100, 101, 102, 103, 104, 105, 106],
            "Close": [100, 101, 102, 103, 104, 105, 106, 107],
            "High": [101, 102, 103, 104, 105, 106, 107, 108],
            "Low": [99, 100, 101, 102, 103, 104, 105, 106],
        }
    )  # No strong pivot points

    mock_finder.source_data_frame = df
    result = mock_finder.find_support_resistance_zones()
    last_row = result.iloc[-1]
    assert np.isnan(last_row["resistance_range"])
    assert np.isnan(last_row["support_range"])


def test_minimal_data(mock_finder):
    """Test that minimal data does not cause errors and returns NaN."""
    df = pd.DataFrame(
        {"Open": [99, 100], "Close": [100, 101], "High": [101, 102], "Low": [99, 100]}
    )  # Too small for meaningful analysis
    mock_finder.source_data_frame = df
    result = mock_finder.find_support_resistance_zones()
    last_row = result.iloc[-1]

    assert np.isnan(last_row["resistance_range"])
    assert np.isnan(last_row["support_range"])
