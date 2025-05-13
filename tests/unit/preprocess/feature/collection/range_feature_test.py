from unittest.mock import MagicMock, patch

import pytest
from pandas import DataFrame, DatetimeIndex, date_range

from drl_trading_framework.preprocess.feature.collection.range_feature import (
    RangeFeature,
)
from drl_trading_framework.preprocess.feature.custom.enum.wick_handle_strategy_enum import (
    WICK_HANDLE_STRATEGY,
)


@pytest.fixture
def mock_data() -> DataFrame:
    # Create sample dates
    dates = date_range(start="2022-01-01", periods=5, freq="D")

    # Create the DataFrame with sample data and datetime index
    data = {
        "Open": [1, 2, 3, 4, 5],
        "High": [2, 3, 4, 5, 6],
        "Low": [0.5, 1.5, 2.5, 3.5, 4.5],
        "Close": [1.5, 2.5, 3.5, 4.5, 5.5],
    }

    df = DataFrame(data, index=dates)
    df.index.name = "Time"
    return df


@pytest.fixture
def config():
    mock_config = MagicMock()
    mock_config.lookback = 5
    mock_config.wick_handle_strategy = WICK_HANDLE_STRATEGY.LAST_WICK_ONLY
    return mock_config


@pytest.fixture
def feature(mock_data, config):
    return RangeFeature(mock_data, config, "test")


@pytest.fixture
def prepared_source_df(feature):
    """Fixture that mocks the _prepare_source_df method and returns a controlled DataFrame."""
    with patch.object(feature, "_prepare_source_df") as mock_prepare:
        mock_df = feature.df_source.copy()
        mock_prepare.return_value = mock_df
        yield mock_df


@patch(
    "drl_trading_framework.preprocess.feature.collection.range_feature.SupportResistanceFinder"
)
def test_compute_range_feature(mock_finder_class, feature, config, prepared_source_df):
    # Given
    mock_finder = MagicMock()
    # Create a DataFrame with the same index as prepared_source_df
    result_data = DataFrame(
        {"resistance_range": [1, 0, 1, 0, 1], "support_range": [0, 1, 0, 1, 0]},
        index=prepared_source_df.index,
    )
    mock_finder.find_support_resistance_zones.return_value = result_data
    mock_finder_class.return_value = mock_finder

    # When
    result_df = feature.compute()

    # Then
    # Add assertions for calls to the mocked class if needed
    # mock_finder_class.assert_called_once_with(...)

    # Check index type and columns exist
    assert isinstance(result_df.index, DatetimeIndex)
    assert "resistance_range5test" in result_df.columns
    assert "support_range5test" in result_df.columns
    assert len(result_df) == 5

    # Check the mocked values are passed through
    assert list(result_df["resistance_range5test"]) == [1, 0, 1, 0, 1]
    assert list(result_df["support_range5test"]) == [0, 1, 0, 1, 0]
