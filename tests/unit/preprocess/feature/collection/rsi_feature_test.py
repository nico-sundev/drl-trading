from unittest.mock import MagicMock, patch

import pytest
from pandas import DataFrame, Series, to_datetime

from ai_trading.preprocess.feature.collection.rsi_feature import RsiFeature


@pytest.fixture
def mock_data() -> DataFrame:
    data = {"Time": [], "Open": [], "High": [], "Low": [], "Close": []}

    # Create the DataFrame
    df = DataFrame(data)

    # Convert Time column to datetime
    df["Time"] = to_datetime(df["Time"])
    return df


@pytest.fixture
def feature(mock_data):
    return RsiFeature(mock_data, "test")


@pytest.fixture
def config():
    mock_config = MagicMock()
    mock_config.length = 14
    return mock_config


@patch("pandas_ta.rsi")
def test_compute_rsi(patched_rsi, feature, config):
    # Given
    patched_rsi.return_value = Series([50, 55, 60, 65, 70, 75])

    # When
    result = feature.compute(config)

    # Then
    patched_rsi.assert_called_once_with(
        feature.df_source["Close"], length=config.length
    )
    assert "Time" in result.columns
    assert "rsi_14test" in result.columns
    expected_values = [50, 55, 60, 65, 70, 75]
    assert result["rsi_14test"].head(6).tolist() == expected_values
