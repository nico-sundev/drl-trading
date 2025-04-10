import json
import tempfile
import pytest
from ai_trading.config.config_loader import ConfigLoader
from ai_trading.config.feature_config_collection import (
    MacdConfig,
    RangeConfig,
    RocConfig,
    RsiConfig,
)
from ai_trading.preprocess.feature.custom.enum.wick_handle_strategy_enum import (
    WICK_HANDLE_STRATEGY,
)


@pytest.fixture
def temp_config_file():
    """Creates a temporary JSON config file for testing."""
    config_data = {
        "localDataImportConfig": {
            "datasets": [
                {
                    "timeframe": "H1",
                    "base_dataset": True,
                    "file_path": "../../resources/test_H1.csv",
                },
                {
                    "timeframe": "H4",
                    "base_dataset": False,
                    "file_path": "../../resources/test_H4.csv",
                },
            ]
        },
        "featuresConfig": {
            "featureDefinitions": [
                {
                    "name": "macd",
                    "enabled": True,
                    "derivatives": [],
                    "parameterSets": [
                        {"enabled": True, "fast": 3, "slow": 5, "signal": 7}
                    ],
                },
                {
                    "name": "rsi",
                    "enabled": True,
                    "derivatives": [1],
                    "parameterSets": [
                        {"enabled": True, "length": 7},
                        {"enabled": True, "length": 14},
                        {"enabled": True, "length": 21},
                    ],
                },
                {
                    "name": "roc",
                    "enabled": True,
                    "derivatives": [1],
                    "parameterSets": [
                        {"enabled": True, "length": 1},
                        {"enabled": True, "length": 3},
                        {"enabled": True, "length": 5},
                    ],
                },
                {
                    "name": "range",
                    "enabled": True,
                    "derivatives": [],
                    "parameterSets": [
                        {
                            "enabled": True,
                            "lookback": 5,
                            "wickHandleStrategy": "LAST_WICK_ONLY",
                        }
                    ],
                },
            ]
        },
        "rlModelConfig": {
            "agents": [],
            "trainingSplitRatio": 0.8,
            "validatingSplitRatio": 0.1,
            "testingSplitRatio": 0.1,
        },
        "environmentConfig": {"fee": 0.005, "slippageAtrBased": 0.01},
    }

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        json.dump(config_data, temp_file)
        temp_file_path = temp_file.name

    yield temp_file_path, config_data  # Yield file path and expected config data

    # Cleanup
    import os

    os.remove(temp_file_path)


def test_load_config_from_json(temp_config_file):
    temp_file_path, expected_data = temp_config_file
    config = ConfigLoader.get_config(temp_file_path)

    # Helper to extract config by name
    def get_feature_param_set(name, index=0):
        for feat in config.features_config.feature_definitions:
            if feat.name == name:
                return feat.parsed_parameter_sets[index]
        return None

    # MACD
    macd = get_feature_param_set("macd")
    assert isinstance(macd, MacdConfig)
    assert macd.fast == 3
    assert macd.slow == 5
    assert macd.signal == 7

    # RSI
    rsi = get_feature_param_set("rsi")
    assert isinstance(rsi, RsiConfig)
    assert rsi.length == 7

    # ROC
    roc = get_feature_param_set("roc", index=2)
    assert isinstance(roc, RocConfig)
    assert roc.length == 5

    # RANGE
    range_cfg = get_feature_param_set("range")
    assert isinstance(range_cfg, RangeConfig)
    assert range_cfg.lookback == 5
    assert range_cfg.wick_handle_strategy == WICK_HANDLE_STRATEGY.LAST_WICK_ONLY
