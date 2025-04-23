from unittest.mock import MagicMock, patch

import pytest
from pandas import DataFrame, date_range

from ai_trading.preprocess.feature.collection.range_feature import RangeFeature
from ai_trading.preprocess.feature.custom.enum.wick_handle_strategy_enum import (
    WICK_HANDLE_STRATEGY,
)


@pytest.fixture
def mock_data() -> DataFrame:
    return DataFrame(
        {
            "Time": date_range(start="2022-01-01", periods=5, freq="D"),
            "Open": [1, 2, 3, 4, 5],
            "High": [2, 3, 4, 5, 6],
            "Low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "Close": [1.5, 2.5, 3.5, 4.5, 5.5],
        }
    )


@pytest.fixture
def config():
    mock_config = MagicMock()
    mock_config.lookback = 5
    mock_config.wick_handle_strategy = WICK_HANDLE_STRATEGY.LAST_WICK_ONLY
    return mock_config


@pytest.fixture
def feature(mock_data, config):
    return RangeFeature(mock_data, config, "test")


@patch("ai_trading.preprocess.feature.collection.range_feature.SupportResistanceFinder")
def test_compute_range_feature(mock_finder_class, mock_data, feature, config):
    # Given
    mock_finder = MagicMock()
    mock_finder.find_support_resistance_zones.return_value = DataFrame(
        {"resistance_range": [1, 0, 1, 0, 1], "support_range": [0, 1, 0, 1, 0]}
    )
    mock_finder_class.return_value = mock_finder

    # When
    result_df = feature.compute()

    # Then
    # mock_finder_class.assert_called_once_with(
    #     feature.df_source["Close"], length=config.length
    # )
    # Check columns exist
    assert "resistance_range5test" in result_df.columns
    assert "support_range5test" in result_df.columns
    assert len(result_df) == 5
    assert result_df["Time"].equals(mock_data["Time"])

    # Check the mocked values are passed through
    assert list(result_df["resistance_range5test"]) == [1, 0, 1, 0, 1]
    assert list(result_df["support_range5test"]) == [0, 1, 0, 1, 0]
