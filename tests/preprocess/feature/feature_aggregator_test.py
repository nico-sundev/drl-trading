
from unittest.mock import MagicMock
from pandas import DataFrame
import pytest

from ai_trading.preprocess.feature.custom.enum.wick_handle_strategy_enum import WICK_HANDLE_STRATEGY
from ai_trading.preprocess.feature.feature_aggregator import FeatureAggregator
from ai_trading.preprocess.feature.feature_factory import FeatureFactory


@pytest.fixture
def mock_config():
    """Fixture to create a mock config object"""
    mock_config = MagicMock()
    mock_config.macd.fast = 3
    mock_config.macd.slow = 6
    mock_config.macd.signal = 5
    mock_config.roc_lengths = [14, 7, 3]
    mock_config.rsi_lengths = [14, 3]
    mock_config.ranges.lookback = 5
    mock_config.ranges.wick_handle_strategy = WICK_HANDLE_STRATEGY.LAST_WICK_ONLY
    return mock_config
    
@pytest.fixture
def feature_factory():
    return MagicMock(spec=FeatureFactory)  # Mocking the FeatureEngine class

@pytest.fixture
def feature_aggregator(mock_config, feature_factory):
    return FeatureAggregator(feature_factory, mock_config)

def test_compute_all_success(feature_factory, feature_aggregator: FeatureAggregator):
    # Given
    feature_factory.compute_macd_signals = MagicMock(return_value=DataFrame({
        "Time": [1, 2, 3],
        "macd_12_26": [0.1, 0.2, 0.3],
        "signal_12_26": [0.01, 0.02, 0.03]
    }))
    feature_factory.compute_roc = MagicMock(return_value=DataFrame({
        "Time": [1, 2, 3],
        "roc_14": [0.5, 0.6, 0.7]
    }))
    feature_factory.compute_rsi = MagicMock(return_value=DataFrame({
        "Time": [1, 2, 3],
        "rsi_14": [50, 55, 60]
    }))
    feature_factory.compute_ranges = MagicMock(return_value=DataFrame({
        "Time": [1, 2, 3],
        "resistance_range5": [0, 0, -10],
        "support_range5": [0, 0, 10]
    }))
    
    # When
    result = feature_aggregator.compute()

    # Then
    feature_factory.compute_macd_signals.assert_called_once()
    feature_factory.compute_roc.assert_called()
    feature_factory.compute_rsi.assert_called()
    feature_factory.compute_ranges.assert_called()

    assert isinstance(result, DataFrame)
    assert "Time" in result.columns
    assert "macd_12_26" in result.columns
    assert "signal_12_26" in result.columns
    assert "roc_14" in result.columns
    assert "rsi_14" in result.columns
    assert "resistance_range5" in result.columns
    assert "support_range5" in result.columns
    assert result.shape[0] == 3  # Ensure correct number of rows