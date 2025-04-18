from unittest.mock import MagicMock, patch

import pytest
from pandas import DataFrame, Series, to_datetime

from ai_trading.preprocess.feature.collection.roc_feature import RocFeature


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
    return RocFeature(mock_data, "test")


@pytest.fixture
def config():
    mock_config = MagicMock()
    mock_config.length = 3
    return mock_config


@patch("pandas_ta.roc")
def test_compute_roc(patched_roc, feature, config):

    # Given
    patched_roc.return_value = Series([50, 55, 60, 65, 70, 75])

    # When
    result = feature.compute(config)

    # Then
    patched_roc.assert_called_once_with(
        feature.df_source["Close"], length=config.length
    )
    assert "Time" in result.columns
    assert "roc_3test" in result.columns
    expected_values = [50, 55, 60, 65, 70, 75]
    assert result["roc_3test"].head(6).tolist() == expected_values
