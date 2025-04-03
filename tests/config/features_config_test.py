import json
import tempfile
import pytest

from ai_trading.config.config_loader import ConfigLoader
from ai_trading.preprocess.feature.feature_config import MACDConfig

@pytest.fixture
def temp_config_file():
    """Creates a temporary JSON config file for testing."""
    config_data = {
        "macd": {"fast": 12, "slow": 26, "signal": 9},
        "rsi_lengths": [14, 3],
        "roc_lengths": [14, 7, 3],
        "range": {"lookback": 5, "wick_handle_strategy": "PREVIOUS_WICK_ONLY"}
    }
    
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        json.dump(config_data, temp_file)
        temp_file_path = temp_file.name
    
    yield temp_file_path, config_data  # Yield file path and expected config data
    
    # Cleanup
    import os
    os.remove(temp_file_path)

def test_load_config_from_json(temp_config_file):
    """Test if the JSON config is loaded correctly."""
    temp_file_path, expected_data = temp_config_file
    config = ConfigLoader.from_json(temp_file_path)

    # Check MACD values
    assert isinstance(config.macd, MACDConfig)
    assert config.macd.fast == expected_data["macd"]["fast"]
    assert config.macd.slow == expected_data["macd"]["slow"]
    assert config.macd.signal == expected_data["macd"]["signal"]

    # Check RSI and ROC lengths
    assert config.rsi_lengths == expected_data["rsi_lengths"]
    assert config.roc_lengths == expected_data["roc_lengths"]
    
    # Check Range indicator values
    assert config.range.lookback == expected_data["range"]["lookback"]
    assert config.range.wick_handle_strategy == expected_data["range"]["wick_handle_strategy"]
