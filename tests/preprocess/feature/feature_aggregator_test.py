
from unittest.mock import MagicMock
from pandas import DataFrame
import pytest

from ai_trading.preprocess.feature.feature_aggregator import FeatureAggregator
from ai_trading.preprocess.feature.feature_engine import FeatureEngine


@pytest.fixture
def mock_config():
    """Fixture to create a mock config object"""
    mock_config = MagicMock()
    mock_config.macd.fast = 3
    mock_config.macd.slow = 6
    mock_config.macd.signal = 5
    mock_config.roc_lengths = [14, 7, 3]
    mock_config.rsi_lengths = [14, 3]
    return mock_config
    
@pytest.fixture
def feature_engine():
    return MagicMock(spec=FeatureEngine)  # Mocking the FeatureEngine class

@pytest.fixture
def feature_aggregator(mock_config, feature_engine):
    return FeatureAggregator(feature_engine, mock_config)

def test_compute_all_success(feature_engine, feature_aggregator: FeatureAggregator):
    # Given
    feature_engine.compute_macd_signals = MagicMock(return_value=DataFrame({
        "Time": [1, 2, 3],
        "macd_12_26": [0.1, 0.2, 0.3],
        "signal_12_26": [0.01, 0.02, 0.03]
    }))
    feature_engine.compute_roc = MagicMock(return_value=DataFrame({
        "Time": [1, 2, 3],
        "roc_14": [0.5, 0.6, 0.7]
    }))
    feature_engine.compute_rsi = MagicMock(return_value=DataFrame({
        "Time": [1, 2, 3],
        "rsi_14": [50, 55, 60]
    }))
    
    # When
    result = feature_aggregator.compute()

    # Then
    feature_engine.compute_macd_signals.assert_called_once()
    feature_engine.compute_roc.assert_called()
    feature_engine.compute_rsi.assert_called()

    assert isinstance(result, DataFrame)
    assert "Time" in result.columns
    assert "macd_12_26" in result.columns
    assert "signal_12_26" in result.columns
    assert "roc_14" in result.columns
    assert "rsi_14" in result.columns
    assert result.shape[0] == 3  # Ensure correct number of rows