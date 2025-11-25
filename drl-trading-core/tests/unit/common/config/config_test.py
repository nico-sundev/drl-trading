
import pytest
from drl_trading_common.config.application_config import ApplicationConfig
from drl_trading_common.config.config_loader import ConfigLoader

from drl_trading_core.core.service.feature_definition_parser import FeatureDefinitionParser

@pytest.fixture
def feature_definition_parser(feature_factory):
    return FeatureDefinitionParser(feature_factory)

def test_load_config_from_json(temp_config_file, feature_factory, mock_rsi_config_class, feature_definition_parser):
    temp_file_path, expected_data = temp_config_file

    # Load config with patched feature factory
    config = ConfigLoader.get_config(ApplicationConfig, temp_file_path)

    # Use the utils.parse_all_parameters function with our mock factory
    feature_definition_parser.parse_feature_definitions(config.features_config.feature_definitions)

    # Helper to extract config by name
    def get_feature_param_set(name, index=0):
        for feat in config.features_config.feature_definitions:
            if feat.name == name:
                return list(feat.parsed_parameter_sets.values())[index]
        return None

    # RSI
    rsi = get_feature_param_set("rsi")
    assert isinstance(rsi, mock_rsi_config_class)
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
    assert features_store_config.cache_enabled is False
    assert features_store_config.config_directory == "testrepo"
    assert features_store_config.local_repo_config.repo_path == "test_data"
    assert features_store_config.entity_name == "symbol"
    assert features_store_config.ttl_days == 365
    assert features_store_config.online_enabled is True

    context_config = config.context_feature_config
    assert context_config.primary_context_columns == ["High", "Low", "Close"]
    assert context_config.derived_context_columns == ["Open", "Volume"]
    assert context_config.optional_context_columns == ["Atr"]
    assert context_config.time_column == "Time"
