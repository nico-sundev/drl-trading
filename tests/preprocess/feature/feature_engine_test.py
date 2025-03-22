from unittest.mock import patch
import numpy as np
from pandas import DataFrame, Series, to_datetime
import pytest

from ai_trading.preprocess.feature.feature_engine import FeatureEngine


@pytest.fixture
def mock_data() -> DataFrame:
    data = {
        "Time": [
        ],
        "Open": [
        ],
        "High": [
        ],
        "Low": [
        ],
        "Close": [
        ]
    }

    # Create the DataFrame
    df = DataFrame(data)

    # Convert Time column to datetime
    df["Time"] = to_datetime(df["Time"])
    return df


@pytest.fixture
def feature_engine(mock_data):
    return FeatureEngine(mock_data, "test")


@patch("pandas_ta.rsi")
def test_compute_rsi(patched_rsi, feature_engine):
    # Given
    rsi_length = 14
    patched_rsi.return_value = Series([50, 55, 60, 65, 70, 75])

    # When
    result = feature_engine.compute_rsi(length=rsi_length)

    # Then
    patched_rsi.assert_called_once_with(
        feature_engine.df_source["Close"], length=rsi_length
    )
    assert "Time" in result.columns
    assert "rsi_14test" in result.columns
    expected_values = [50, 55, 60, 65, 70, 75]
    assert result["rsi_14test"].head(6).tolist() == expected_values


@patch("pandas_ta.roc")
def test_compute_roc(patched_roc, feature_engine):

    # Given
    roc_length = 3
    patched_roc.return_value = Series([50, 55, 60, 65, 70, 75])

    # When
    result = feature_engine.compute_roc(roc_length)

    # Then
    patched_roc.assert_called_once_with(
        feature_engine.df_source["Close"], length=roc_length
    )
    assert "Time" in result.columns
    assert f"roc_{roc_length}test" in result.columns
    expected_values = [50, 55, 60, 65, 70, 75]
    assert result[f"roc_{roc_length}test"].head(6).tolist() == expected_values

@patch("pandas_ta.macd")
def test_compute_macd_signals(patched_macd, feature_engine):

    # Given
    fast_length = 3
    slow_length = 6
    signal_length = 5
    macd_result = DataFrame()
    macd_result[f"MACD_{fast_length}_{slow_length}_{signal_length}_A_0"] = Series(
        [0, 0, 1, 0, 1, 0] # MACD Trend
    )
    macd_result[f"MACDh_{fast_length}_{slow_length}_{signal_length}_XB_0"] = Series(
        [1, 0, 1, 0, 1, 1] # Cross bearish
    )
    macd_result[f"MACDh_{fast_length}_{slow_length}_{signal_length}_XA_0"] = Series(
        [1, 1, 1, 1, 1, 1] # Cross bullish
    )
    patched_macd.return_value = macd_result

    # When
    result = feature_engine.compute_macd_signals(
        fast_length, slow_length, signal_length
    )

    # Then
    patched_macd.assert_called_once_with(
        feature_engine.df_source["Close"],
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
