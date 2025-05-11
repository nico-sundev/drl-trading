from unittest.mock import MagicMock, patch

import pytest
from pandas import DataFrame, DatetimeIndex, Series, to_datetime

from ai_trading.preprocess.feature.collection.rsi_feature import RsiFeature


@pytest.fixture
def mock_data() -> DataFrame:
    # Create sample dates
    dates = to_datetime(
        [
            "2023-01-01",
            "2023-01-02",
            "2023-01-03",
            "2023-01-04",
            "2023-01-05",
            "2023-01-06",
        ]
    )

    # Create the DataFrame with sample data and datetime index
    data = {
        "Open": [100, 101, 102, 103, 104, 105],
        "High": [105, 106, 107, 108, 109, 110],
        "Low": [95, 96, 97, 98, 99, 100],
        "Close": [103, 102, 104, 105, 107, 106],
    }

    df = DataFrame(data, index=dates)
    df.index.name = "Time"
    return df


@pytest.fixture
def config():
    mock_config = MagicMock()
    mock_config.length = 14
    return mock_config


@pytest.fixture
def feature(mock_data, config):
    return RsiFeature(mock_data, config, "test")


@patch("pandas_ta.rsi")
def test_compute_rsi(patched_rsi, feature, config):
    # Given
    expected_values = [50, 55, 60, 65, 70, 75]

    # Mock the _prepare_source_df method to return a known value
    with patch.object(feature, "_prepare_source_df") as mock_prepare:
        mock_df = feature.df_source.copy()
        mock_prepare.return_value = mock_df

        # Create a Series with the same index as mock_df for the RSI values
        rsi_series = Series(expected_values, index=mock_df.index)
        patched_rsi.return_value = rsi_series

        # When
        result = feature.compute()

        # Then
        patched_rsi.assert_called_once_with(mock_df["Close"], length=config.length)

    assert isinstance(result.index, DatetimeIndex)
    assert "rsi_14test" in result.columns
    assert result["rsi_14test"].head(6).tolist() == expected_values
