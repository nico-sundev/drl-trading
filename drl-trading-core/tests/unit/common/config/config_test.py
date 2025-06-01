from typing import Literal
from unittest.mock import MagicMock

import pytest
from drl_trading_common import BaseParameterSetConfig
from drl_trading_common.config.config_loader import ConfigLoader

from drl_trading_core.common.config.utils import parse_all_parameters
from drl_trading_core.preprocess.feature.feature_factory import (
    FeatureFactoryInterface,
)


class RsiConfig(BaseParameterSetConfig):
    type: Literal["rsi"]
    length: int


@pytest.fixture
def mock_feature_factory():
    """Creates a mock feature factory for testing."""
    mock_factory = MagicMock(spec=FeatureFactoryInterface)

    # Configure mock factory to return appropriate config classes
    def get_config_class_side_effect(feature_name):
        feature_name = feature_name.lower()
        config_classes = {
            "rsi": RsiConfig,
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


def test_load_config_from_json(temp_config_file, mock_feature_factory):
    temp_file_path, expected_data = temp_config_file

    # Load config with patched feature factory
    config = ConfigLoader.get_config(temp_file_path)

    # Use the utils.parse_all_parameters function with our mock factory
    parse_all_parameters(config.features_config.feature_definitions, mock_feature_factory)

    # Helper to extract config by name
    def get_feature_param_set(name, index=0):
        for feat in config.features_config.feature_definitions:
            if feat.name == name:
                return feat.parsed_parameter_sets[index]
        return None

    # RSI
    rsi = get_feature_param_set("rsi")
    assert isinstance(rsi, RsiConfig)
    assert rsi.length == 7

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

    context_config = config.context_feature_config
    assert context_config.primary_context_columns == ["High", "Low", "Close"]
    assert context_config.derived_context_columns == ["Open", "Volume"]
    assert context_config.optional_context_columns == ["Atr"]
    assert context_config.time_column == "Time"
