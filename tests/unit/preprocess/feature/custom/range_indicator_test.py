from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from ai_trading.preprocess.feature.custom.enum.wick_handle_strategy_enum import (
    WICK_HANDLE_STRATEGY,
)
from ai_trading.preprocess.feature.custom.range_indicator import (
    PIVOT_HIGH,
    PIVOT_LOW,
    SupportResistanceFinder,
)


@pytest.fixture
def mock_data() -> DataFrame:
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
def mock_pivot_cache() -> DataFrame:
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
def mock_finder(mock_data: DataFrame) -> SupportResistanceFinder:
    """Returns an instance of SupportResistanceFinder with a small lookback."""
    return SupportResistanceFinder(mock_data, 5, WICK_HANDLE_STRATEGY.LAST_WICK_ONLY)


def test_validate_dataframe_success(mock_finder: SupportResistanceFinder) -> None:
    """Test that validation passes for a valid DataFrame."""
    mock_finder.validate_dataframe()


def test_validate_dataframe_missing_columns(
    mock_finder: SupportResistanceFinder,
) -> None:
    """Test that validation fails when required columns are missing."""
    df = pd.DataFrame({"Close": [100, 101, 102]})  # Missing "High" and "Low"
    mock_finder.source_data_frame = df
    with pytest.raises(ValueError, match="Missing required columns"):
        mock_finder.validate_dataframe()


def test_validate_dataframe_empty(mock_finder: SupportResistanceFinder) -> None:
    """Test that validation fails on an empty DataFrame."""
    df = pd.DataFrame(columns=["Close", "High", "Low"])
    mock_finder.source_data_frame = df
    with pytest.raises(ValueError, match="DataFrame is empty"):
        mock_finder.validate_dataframe()


def test_validate_dataframe_nan_values(mock_finder: SupportResistanceFinder) -> None:
    """Test that validation fails when NaN values are present."""
    df = pd.DataFrame(
        {"Close": [100, np.nan, 102], "High": [101, 103, 102], "Low": [99, 100, 100]}
    )
    mock_finder.source_data_frame = df
    with pytest.raises(ValueError, match="DataFrame contains NaN values"):
        mock_finder.validate_dataframe()


def test_find_pivot_points(
    mock_finder: SupportResistanceFinder, mock_data: DataFrame
) -> None:
    """Test that pivot highs and lows are correctly identified."""
    expected_highs: List[bool] = [
        False,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        False,
    ]
    expected_lows: List[bool] = [
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]

    for index, _row in mock_data.iterrows():
        pivot_points: Tuple[bool, bool] = mock_finder.find_pivot_points(index)
        assert pivot_points[0] == expected_highs[index]
        assert pivot_points[1] == expected_lows[index]


def test_clean_pivot_cache(
    mock_finder: SupportResistanceFinder,
    mock_data: DataFrame,
    mock_pivot_cache: DataFrame,
) -> None:
    """Test that old pivot points are removed from cache."""
    last_close = mock_data.iloc[-1]["Close"]
    mock_finder.pivot_cache = mock_pivot_cache
    assert len(mock_finder.pivot_cache) == 3
    mock_finder.clean_pivot_cache(last_close)
    assert len(mock_finder.pivot_cache) == 2


def test_find_next_zone(
    mock_finder: SupportResistanceFinder,
    mock_data: DataFrame,
    mock_pivot_cache: DataFrame,
) -> None:
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


def test_find_support_resistance_zones(mock_finder: SupportResistanceFinder) -> None:
    """Test the full support & resistance zone detection process."""
    result = mock_finder.find_support_resistance_zones()

    assert "support_range" in result.columns
    assert "resistance_range" in result.columns

    last_row = result.iloc[-1]
    assert not np.isnan(last_row["support_range"]) or not np.isnan(
        last_row["resistance_range"]
    )


def test_no_zone_found_within_lookback(mock_finder: SupportResistanceFinder) -> None:
    """Test that NaN is returned when no support/resistance is found in lookback range."""
    mock_finder.pivot_cache = pd.DataFrame(columns=["index", "top", "bottom", "type"])
    last_close = 100.0

    resistance = mock_finder.find_next_zone(PIVOT_HIGH, last_close)
    support = mock_finder.find_next_zone(PIVOT_LOW, last_close)

    assert np.isnan(resistance["top"]) and np.isnan(resistance["bottom"])
    assert np.isnan(support["top"]) and np.isnan(support["bottom"])


def test_minimal_data(mock_finder: SupportResistanceFinder) -> None:
    """Test that minimal data does not cause errors and returns NaN."""
    df = pd.DataFrame(
        {
            "Open": [100],
            "Close": [101],
            "High": [102],
            "Low": [99],
        },
        index=[0],
    )
    finder = SupportResistanceFinder(df, 5, WICK_HANDLE_STRATEGY.LAST_WICK_ONLY)
    result = finder.find_support_resistance_zones()

    assert "support_range" in result.columns
    assert "resistance_range" in result.columns
    assert np.isnan(result.iloc[0]["support_range"])
    assert np.isnan(result.iloc[0]["resistance_range"])
