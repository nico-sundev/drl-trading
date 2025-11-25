import json
import tempfile

import pytest


@pytest.fixture
def temp_config_file():
    """Creates a temporary JSON config file for testing."""
    config_data = {
        "app_name": "drl-trading-core",
        "version": "1.0.0",
        "stage": "local",
        "local_data_import_config": {
            "symbols": [
                {
                    "symbol": "EURUSD",
                    "datasets": [
                        {
                            "timeframe": "1h",
                            "base_dataset": True,
                            "file_path": "../../resources/test_H1.csv",
                        },
                        {
                            "timeframe": "1h",
                            "base_dataset": False,
                            "file_path": "../../resources/test_H4.csv",
                        },
                    ],
                }
            ],
            "limit": 100,
            "strategy": "csv",
        },
        "features_config": {
            "dataset_definitions": {"EURUSD": ["1h", "4h"], "BTCUSDT": ["1h", "4h"]},
            "feature_definitions": [
                {
                    "name": "rsi",
                    "enabled": True,
                    "derivatives": [1],
                    "parameter_sets": [
                        {"enabled": True, "length": 7},
                        {"enabled": True, "length": 14},
                        {"enabled": True, "length": 21},
                    ],
                }
            ],
        },
        "rl_model_config": {
            "agents": ["PPO", "A2C", "DDPG", "SAC", "TD3", "Ensemble"],
            "training_split_ratio": 0.8,
            "validating_split_ratio": 0.1,
            "testing_split_ratio": 0.1,
            "agent_threshold": 0.1,
            "total_timesteps": 10000,
        },
        "environment_config": {
            "fee": 0.005,
            "slippage_atr_based": 0.01,
            "slippage_against_trade_probability": 0.6,
            "start_balance": 10000.0,
            "max_daily_drawdown": 0.02,
            "max_alltime_drawdown": 0.05,
            "max_percentage_open_position": 100.0,
            "min_percentage_open_position": 1.0,
            "max_time_in_trade": 10,
            "optimal_exit_time": 3,
            "variance_penalty_weight": 0.5,
            "atr_penalty_weight": 0.3,
        },
        "feature_store_config": {
            "cache_enabled": False,
            "config_directory": "testrepo",
            "entity_name": "symbol",
            "ttl_days": 365,
            "online_enabled": True,
            "service_name": "test_service",
            "service_version": "1.0.0",
            "offline_repo_strategy": "local",
            "local_repo_config": {
                "repo_path": "test_data"
            },
            "s3_repo_config": {
                "bucket_name": "drl-trading-features-test",
                "prefix": "features",
                "endpoint_url": None,
                "region": "us-east-1",
                "access_key_id": None,
                "secret_access_key": None
            }
        },
        "context_feature_config": {
            "primary_context_columns": ["High", "Low", "Close"],
            "derived_context_columns": ["Open", "Volume"],
            "optional_context_columns": ["Atr"],
            "time_column": "Time",
        },
    }

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        json.dump(config_data, temp_file)
        temp_file_path = temp_file.name

    yield temp_file_path, config_data  # Yield file path and expected config data

    # Cleanup
    import os

    os.remove(temp_file_path)
