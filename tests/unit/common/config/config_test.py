import json
import tempfile
from unittest.mock import MagicMock

import pytest

from ai_trading.common.config.config_loader import ConfigLoader
from ai_trading.common.config.feature_config_collection import (
    MacdConfig,
    RangeConfig,
    RocConfig,
    RsiConfig,
)
from ai_trading.common.config.feature_config_factory import FeatureConfigFactory
from ai_trading.preprocess.feature.custom.enum.wick_handle_strategy_enum import (
    WICK_HANDLE_STRATEGY,
)


@pytest.fixture
def mock_feature_config_factory():
    """Creates a mock feature config factory for testing."""
    mock_factory = MagicMock(spec=FeatureConfigFactory)

    # Configure mock factory to return appropriate config classes
    def get_config_class_side_effect(feature_name):
        feature_name = feature_name.lower()
        config_classes = {
            "macd": MacdConfig,
            "rsi": RsiConfig,
            "roc": RocConfig,
            "range": RangeConfig,
            "rvi": RsiConfig,  # Add any other features used in tests
        }
        return config_classes.get(feature_name)

    mock_factory.get_config_class.side_effect = get_config_class_side_effect

    # Configure create_config_instance to return proper config objects
    def create_config_instance_side_effect(feature_name, config_data):
        config_class = mock_factory.get_config_class(feature_name)
        if not config_class:
            return None

        # Add type field for discriminated unions if not present
        instance_data = config_data.copy()
        if "type" not in instance_data:
            instance_data["type"] = feature_name.lower()

        return config_class(**instance_data)

    mock_factory.create_config_instance.side_effect = create_config_instance_side_effect

    return mock_factory


@pytest.fixture
def temp_config_file():
    """Creates a temporary JSON config file for testing."""
    config_data = {
        "localDataImportConfig": {
            "symbols": [
                {
                    "symbol": "EURUSD",
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
                    ],
                }
            ],
            "limit": 100,
        },
        "featuresConfig": {
            "featureDefinitions": [
                {
                    "name": "macd",
                    "enabled": True,
                    "derivatives": [],
                    "parameterSets": [
                        {
                            "enabled": True,
                            "fast_length": 3,
                            "slow_length": 5,
                            "signal_length": 7,
                        }
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
            "agentRegistryPackage": "ai_trading.common.agents",
            "agents": ["PPO", "A2C", "DDPG", "SAC", "TD3", "Ensemble"],
            "trainingSplitRatio": 0.8,
            "validatingSplitRatio": 0.1,
            "testingSplitRatio": 0.1,
            "agent_threshold": 0.1,
            "total_timesteps": 10000,
        },
        "environmentConfig": {
            "fee": 0.005,
            "slippageAtrBased": 0.01,
            "slippageAgainstTradeProbability": 0.6,
            "startBalance": 10000.0,
            "maxDailyDrawdown": 0.02,
            "maxAlltimeDrawdown": 0.05,
            "maxPercentageOpenPosition": 100.0,
            "minPercentageOpenPosition": 1.0,
            "maxTimeInTrade": 10,
            "optimalExitTime": 3,
            "variancePenaltyWeight": 0.5,
            "atrPenaltyWeight": 0.3,
        },
        "featureStoreConfig": {
            "enabled": False,
            "repo_path": "testrepo",
            "offline_store_path": "test",
            "entity_name": "symbol",
            "ttl_days": 365,
            "online_enabled": True,
        },
    }

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        json.dump(config_data, temp_file)
        temp_file_path = temp_file.name

    yield temp_file_path, config_data  # Yield file path and expected config data

    # Cleanup
    import os

    os.remove(temp_file_path)


def test_load_config_from_json(temp_config_file, mock_feature_config_factory):
    temp_file_path, expected_data = temp_config_file

    # Load config with patched feature config factory
    config = ConfigLoader.get_config(temp_file_path)

    # Use the mock factory for parsing parameters
    config.features_config.parse_all_parameters(mock_feature_config_factory)

    # Helper to extract config by name
    def get_feature_param_set(name, index=0):
        for feat in config.features_config.feature_definitions:
            if feat.name == name:
                return feat.parsed_parameter_sets[index]
        return None

    # MACD
    macd = get_feature_param_set("macd")
    assert isinstance(macd, MacdConfig)
    assert macd.fast_length == 3
    assert macd.slow_length == 5
    assert macd.signal_length == 7

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

    # Verify environment config parameters
    env_config = config.environment_config
    assert env_config.fee == 0.005
    assert env_config.slippage_atr_based == 0.01
    assert env_config.slippage_against_trade_probability == 0.6
    assert env_config.max_time_in_trade == 10
    assert env_config.optimal_exit_time == 3
    assert env_config.variance_penalty_weight == 0.5
    assert env_config.atr_penalty_weight == 0.3
    assert env_config.variance_penalty_weight == 0.5
    assert env_config.optimal_exit_time == 3
    assert env_config.max_time_in_trade == 10

    features_store_config = config.feature_store_config
    assert features_store_config.enabled is False
    assert features_store_config.repo_path == "testrepo"
    assert features_store_config.offline_store_path == "test"
    assert features_store_config.entity_name == "symbol"
    assert features_store_config.ttl_days == 365
    assert features_store_config.online_enabled is True
