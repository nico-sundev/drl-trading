from unittest.mock import MagicMock
import pytest
import pandas as pd
import numpy as np

from ai_trading.preprocess.feature.custom.range_indicator import SupportResistanceFinder


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
    return mock_config

@pytest.fixture
def mock_data():
    """Creates a small mock DataFrame for testing support and resistance."""
    return pd.DataFrame(
        {
            "Open": [99, 100, 102, 101, 105, 103, 108, 106],
            "High": [101, 103, 102, 107, 104, 110, 107, 113],
            "Low": [99, 100, 100, 104, 102, 107, 104, 109],
            "Close": [100, 102, 101, 105, 103, 108, 106, 110]
        }
    )


@pytest.fixture
def mock_finder(mock_config):
    """Returns an instance of SupportResistanceFinder with a small lookback."""
    return SupportResistanceFinder(mock_config)


def test_validate_dataframe_success(mock_finder, mock_data):
    """Test that validation passes for a valid DataFrame."""
    mock_finder.validate_dataframe(mock_data)


def test_validate_dataframe_missing_columns(mock_finder):
    """Test that validation fails when required columns are missing."""
    df = pd.DataFrame({"Close": [100, 101, 102]})  # Missing "High" and "Low"
    with pytest.raises(ValueError, match="Missing required columns"):
        mock_finder.validate_dataframe(df)


def test_validate_dataframe_empty(mock_finder):
    """Test that validation fails on an empty DataFrame."""
    df = pd.DataFrame(columns=["Close", "High", "Low"])
    with pytest.raises(ValueError, match="DataFrame is empty"):
        mock_finder.validate_dataframe(df)


def test_validate_dataframe_nan_values(mock_finder):
    """Test that validation fails when NaN values are present."""
    df = pd.DataFrame(
        {"Close": [100, np.nan, 102], "High": [101, 103, 102], "Low": [99, 100, 100]}
    )
    with pytest.raises(ValueError, match="DataFrame contains NaN values"):
        mock_finder.validate_dataframe(df)


def test_find_pivot_points(mock_finder, mock_data):
    """Test that pivot highs and lows are correctly identified."""
    result = mock_finder.find_pivot_points(mock_data)
    expected_highs = [False, False, False, True, False, True, False, True]
    expected_lows = [False, False, False, False, True, False, True, False]
    print(mock_finder.find_pivot_points.__code__)
    assert list(result["pivot_high"]) == expected_highs
    assert list(result["pivot_low"]) == expected_lows


def test_find_next_support_resistance(mock_finder, mock_data):
    """Test that the correct support and resistance levels are found."""
    last_close = mock_data.iloc[-1]["Close"]
    df = mock_finder.find_pivot_points(mock_data)
    support, resistance = mock_finder.find_next_support_resistance(df, last_close)

    # Check if valid support & resistance zones are found
    assert not np.isnan(support["low"])
    assert not np.isnan(resistance["high"])


def test_cache_persistence(mock_finder, mock_data):
    """Ensure cached zones persist when still valid."""
    last_close = mock_data.iloc[-1]["Close"]
    mock_finder.prev_support = {"low": 102, "high": 104}
    mock_finder.prev_resistance = {"low": 107, "high": 110}

    support, resistance = mock_finder.find_next_support_resistance(
        mock_data, last_close
    )

    # If previous zones are still valid, they should remain unchanged
    assert support == {"low": 102, "high": 104}
    assert resistance == {"low": 107, "high": 110}


def test_zone_reset_when_broken(mock_finder, mock_data):
    """Ensure cached zones are reset when price action breaks them."""
    mock_finder.prev_support = {"low": 102, "high": 104}
    mock_finder.prev_resistance = {"low": 107, "high": 110}

    # Simulate a close that breaks previous resistance
    last_close = 111
    support, resistance = mock_finder.find_next_support_resistance(
        mock_data, last_close
    )

    # Resistance should reset, but support remains
    assert support == {"low": 102, "high": 104}
    assert resistance == {
        "low": np.nan,
        "high": np.nan,
    }  # Reset since resistance was broken


def test_calculate_wick_threshold():
    """Test that the wick threshold calculation is correct."""
    close, extreme = 100, 110
    assert (
        SupportResistanceFinder.calculate_wick_threshold(close, extreme, ratio=0.5)
        == 105
    )

    close = 100
    extreme = 110
    expected_value = 106.67
    ratio = 2 / 3
    tolerance_percentage = 0.02  # 2% tolerance

    result = SupportResistanceFinder.calculate_wick_threshold(close, extreme, ratio)

    # Calculate the acceptable range based on percentage tolerance
    lower_bound = expected_value * (1 - tolerance_percentage)
    upper_bound = expected_value * (1 + tolerance_percentage)

    # Assert that the result is within the tolerance percentage range
    assert (
        lower_bound <= result <= upper_bound
    ), f"Expected {expected_value} ± {tolerance_percentage*100}%, but got {result}"


def test_find_support_resistance_zones(mock_finder, mock_data):
    """Test the full support & resistance zone detection process."""
    result = mock_finder.find_support_resistance_zones(mock_data)

    assert "support_zone_low" in result.columns
    assert "support_zone_high" in result.columns
    assert "resistance_zone_low" in result.columns
    assert "resistance_zone_high" in result.columns

    last_row = result.iloc[-1]
    assert not np.isnan(last_row["support_zone_low"]) or not np.isnan(
        last_row["resistance_zone_high"]
    )


def test_no_zone_found_within_lookback(mock_finder):
    """Test that NaN is returned when no support/resistance is found in the lookback range."""
    df = pd.DataFrame(
        {
            "Close": [100, 101, 102, 103, 104, 105, 106, 107],
            "High": [101, 102, 103, 104, 105, 106, 107, 108],
            "Low": [99, 100, 101, 102, 103, 104, 105, 106],
        }
    )  # No strong pivot points

    result = mock_finder.find_support_resistance_zones(df)
    last_row = result.iloc[-1]

    assert np.isnan(last_row["support_zone_low"])
    assert np.isnan(last_row["resistance_zone_high"])


def test_minimal_data(mock_finder):
    """Test that minimal data does not cause errors and returns NaN."""
    df = pd.DataFrame(
        {"Close": [100, 101], "High": [101, 102], "Low": [99, 100]}
    )  # Too small for meaningful analysis

    result = mock_finder.find_support_resistance_zones(df)
    last_row = result.iloc[-1]

    assert np.isnan(last_row["support_zone_low"])
    assert np.isnan(last_row["resistance_zone_high"])
