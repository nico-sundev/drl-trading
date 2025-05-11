from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pandas import DataFrame, DatetimeIndex, Series, to_datetime

from ai_trading.preprocess.feature.collection.macd_feature import MacdFeature


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
    mock_config.fast_length = 3
    mock_config.slow_length = 6
    mock_config.signal_length = 5
    return mock_config


@pytest.fixture
def feature(mock_data, config):
    return MacdFeature(mock_data, config, "test")


@pytest.fixture
def prepared_source_df(feature):
    """Fixture that mocks the _prepare_source_df method and returns a controlled DataFrame."""
    with patch.object(feature, "_prepare_source_df") as mock_prepare:
        mock_df = feature.df_source.copy()
        mock_prepare.return_value = mock_df
        yield mock_df


@patch("pandas_ta.macd")
def test_compute_macd_signals(patched_macd, feature, config, prepared_source_df):
    # Given
    fast_length = config.fast_length
    slow_length = config.slow_length
    signal_length = config.signal_length
    macd_result = DataFrame(index=prepared_source_df.index)

    # Create Series with the same index as prepared_source_df
    macd_result[f"MACD_{fast_length}_{slow_length}_{signal_length}_A_0"] = Series(
        [0, 0, 1, 0, 1, 0], index=prepared_source_df.index  # MACD Trend
    )
    macd_result[f"MACDh_{fast_length}_{slow_length}_{signal_length}_XB_0"] = Series(
        [1, 0, 1, 0, 1, 1], index=prepared_source_df.index  # Cross bearish
    )
    macd_result[f"MACDh_{fast_length}_{slow_length}_{signal_length}_XA_0"] = Series(
        [1, 1, 1, 1, 1, 1], index=prepared_source_df.index  # Cross bullish
    )
    patched_macd.return_value = macd_result

    # When
    result = feature.compute()

    # Then
    patched_macd.assert_called_once_with(
        prepared_source_df["Close"],
        fast=fast_length,
        slow=slow_length,
        signal=signal_length,
        fillna=np.nan,
        signal_indicators=True,
    )

    assert isinstance(result.index, DatetimeIndex)
    assert "macd_trendtest" in result.columns
    assert "macd_cross_bullishtest" in result.columns
    assert "macd_cross_bearishtest" in result.columns
    assert result["macd_trendtest"].head(6).tolist() == [0, 0, 1, 0, 1, 0]
    assert result["macd_cross_bullishtest"].head(6).tolist() == [1, 1, 1, 1, 1, 1]
    assert result["macd_cross_bearishtest"].head(6).tolist() == [1, 0, 1, 0, 1, 1]
