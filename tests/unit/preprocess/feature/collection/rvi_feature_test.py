from unittest.mock import MagicMock, patch

import pytest
from pandas import DataFrame, Series, to_datetime

from ai_trading.preprocess.feature.collection.rvi_feature import RviFeature


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
    return RviFeature(mock_data, "test")


@pytest.fixture
def config():
    mock_config = MagicMock()
    mock_config.length = 14
    return mock_config


@patch("pandas_ta.rvi")
def test_compute_rvi(patched_rvi, feature, config):
    # Given
    patched_rvi.return_value = Series([50, 55, 60, 65, 70, 75])

    # When
    result = feature.compute(config)

    # Then
    patched_rvi.assert_called_once_with(
        feature.df_source["Close"],
        feature.df_source["High"],
        feature.df_source["Low"],
        length=config.length,
    )
    assert "Time" in result.columns
    assert "rvi_14test" in result.columns
    expected_values = [50, 55, 60, 65, 70, 75]
    assert result["rvi_14test"].head(6).tolist() == expected_values
