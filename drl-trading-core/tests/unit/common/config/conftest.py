import json
import tempfile

import pytest


@pytest.fixture
def temp_config_file():
    """Creates a temporary JSON config file for testing."""
    config_data = {
        "appName": "drl-trading-core",
        "localDataImportConfig": {
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
        "featuresConfig": {
            "datasetDefinitions": {"EURUSD": ["1h", "4h"], "BTCUSDT": ["1h", "4h"]},
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
            ],
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
            "configDirectory": "testrepo",
            "entityName": "symbol",
            "ttlDays": 365,
            "onlineEnabled": True,
            "serviceName": "test_service",
            "serviceVersion": "1.0.0",
            "offlineRepoStrategy": "local",
            "localRepoConfig": {
                "repoPath": "test_data"
            },
            "s3RepoConfig": {
                "bucketName": "drl-trading-features-test",
                "prefix": "features",
                "endpointUrl": None,
                "region": "us-east-1",
                "accessKeyId": None,
                "secretAccessKey": None
            }
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
