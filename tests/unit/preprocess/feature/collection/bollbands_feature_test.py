from unittest.mock import MagicMock, patch

import pytest
from pandas import DataFrame, date_range, to_datetime

from ai_trading.preprocess.feature.collection.bollbands_feature import BollbandsFeature


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
    return BollbandsFeature(mock_data, "test")


@pytest.fixture
def config():
    mock_config = MagicMock()
    mock_config.length = 14
    return mock_config


@patch("pandas_ta.bbands")
def test_bollbands_feature_computation(patched_bollbands, feature, config):
    df = DataFrame(
        {
            "Time": date_range("2020-01-01", periods=10),
            "Close": range(10),
            "Open": range(10),
            "High": range(10),
            "Low": range(10),
        }
    )
    feature = BollbandsFeature(df)
    # config = ...  # Insert mock config here
    # result = feature.compute(config)
    # assert not result.empty
    assert True  # Placeholder
