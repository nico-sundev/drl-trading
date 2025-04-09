import pytest
import pandas as pd
from ai_trading.preprocess.feature.custom.enum.wick_handle_strategy_enum import (
    WICK_HANDLE_STRATEGY,
)
from ai_trading.preprocess.feature.custom.wick_handler import WickHandler


def test_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame is empty"):
        WickHandler.calculate_wick_threshold(df, 0, WICK_HANDLE_STRATEGY.LAST_WICK_ONLY)


def test_invalid_index():
    df = pd.DataFrame({"Open": [100], "Close": [110], "Low": [95], "High": [115]})
    with pytest.raises(IndexError):
        WickHandler.calculate_wick_threshold(df, 1, WICK_HANDLE_STRATEGY.LAST_WICK_ONLY)


def test_last_wick_only():
    df = pd.DataFrame(
        {
            "Open": [100, 105],
            "Close": [110, 107],
            "Low": [95, 102],
            "High": [115, 108],
        }
    )
    assert (
        WickHandler.calculate_wick_threshold(df, 1, WICK_HANDLE_STRATEGY.LAST_WICK_ONLY)
        == 102
    )


def test_previous_wick_only():
    df = pd.DataFrame(
        {
            "Open": [100, 105],
            "Close": [110, 107],
            "Low": [95, 102],
            "High": [115, 108],
        }
    )
    assert (
        WickHandler.calculate_wick_threshold(
            df, 1, WICK_HANDLE_STRATEGY.PREVIOUS_WICK_ONLY
        )
        == 95
    )


def test_mean_wick():
    df = pd.DataFrame(
        {
            "Open": [100, 105],
            "Close": [110, 107],
            "Low": [95, 102],
            "High": [115, 108],
        }
    )
    assert WickHandler.calculate_wick_threshold(
        df, 1, WICK_HANDLE_STRATEGY.MEAN
    ) == pytest.approx((95 + 102) / 2)


def test_max_below_atr():
    df = pd.DataFrame(
        {
            "Open": [100, 105],
            "Close": [110, 107],
            "Low": [95, 102],
            "High": [115, 108],
        }
    )
    atr = 3
    assert WickHandler.calculate_wick_threshold(
        df, 1, WICK_HANDLE_STRATEGY.MAX_BELOW_ATR, atr
    ) == pytest.approx(max(min(95, 102), 105 - 3))


def test_max_below_atr_no_atr():
    df = pd.DataFrame(
        {
            "Open": [100, 105],
            "Close": [110, 107],
            "Low": [95, 102],
            "High": [115, 108],
        }
    )
    with pytest.raises(
        ValueError, match="ATR value is required for MAX_BELOW_ATR strategy"
    ):
        WickHandler.calculate_wick_threshold(df, 1, WICK_HANDLE_STRATEGY.MAX_BELOW_ATR)
