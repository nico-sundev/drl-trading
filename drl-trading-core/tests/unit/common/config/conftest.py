
import json
import tempfile

import pytest


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
            "strategy": "csv",
        },
        "featuresConfig": {
            "featureDefinitions": [
                {
                    "name": "rsi",
                    "enabled": True,
                    "derivatives": [1],
                    "parameterSets": [
                        {"enabled": True, "length": 7},
                        {"enabled": True, "length": 14},
                        {"enabled": True, "length": 21},
                    ],
                }
            ]
        },
        "rlModelConfig": {
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
        "contextFeatureConfig": {
            "primaryContextColumns": ["High", "Low", "Close"],
            "derivedContextColumns": ["Open", "Volume"],
            "optionalContextColumns": ["Atr"],
            "timeColumn": "Time",
        },
    }

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        json.dump(config_data, temp_file)
        temp_file_path = temp_file.name

    yield temp_file_path, config_data  # Yield file path and expected config data

    # Cleanup
    import os

    os.remove(temp_file_path)
