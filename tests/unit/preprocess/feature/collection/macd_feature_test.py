from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pandas import DataFrame, Series, to_datetime

from ai_trading.preprocess.feature.collection.macd_feature import MacdFeature


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
    return MacdFeature(mock_data, "test")


@pytest.fixture
def config():
    mock_config = MagicMock()
    mock_config.fast = 3
    mock_config.slow = 6
    mock_config.signal = 5
    return mock_config


@patch("pandas_ta.macd")
def test_compute_macd_signals(patched_macd, feature, config):

    # Given
    fast_length = config.fast
    slow_length = config.slow
    signal_length = config.signal
    macd_result = DataFrame()
    macd_result[f"MACD_{fast_length}_{slow_length}_{signal_length}_A_0"] = Series(
        [0, 0, 1, 0, 1, 0]  # MACD Trend
    )
    macd_result[f"MACDh_{fast_length}_{slow_length}_{signal_length}_XB_0"] = Series(
        [1, 0, 1, 0, 1, 1]  # Cross bearish
    )
    macd_result[f"MACDh_{fast_length}_{slow_length}_{signal_length}_XA_0"] = Series(
        [1, 1, 1, 1, 1, 1]  # Cross bullish
    )
    patched_macd.return_value = macd_result

    # When
    result = feature.compute(config)

    # Then
    patched_macd.assert_called_once_with(
        feature.df_source["Close"],
        fast=fast_length,
        slow=slow_length,
        signal=signal_length,
        fillna=np.nan,
        signal_indicators=True,
    )
    assert "Time" in result.columns
    assert "macd_trendtest" in result.columns
    assert "macd_cross_bullishtest" in result.columns
    assert "macd_cross_bearishtest" in result.columns
    assert result["macd_trendtest"].head(6).tolist() == [0, 0, 1, 0, 1, 0]
    assert result["macd_cross_bullishtest"].head(6).tolist() == [1, 1, 1, 1, 1, 1]
    assert result["macd_cross_bearishtest"].head(6).tolist() == [1, 0, 1, 0, 1, 1]
