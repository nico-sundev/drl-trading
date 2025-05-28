import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, Optional

import pandas_ta as ta
import pytest
from drl_trading_common.config.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_common.config.config_loader import ConfigLoader
from feast import FeatureStore
from pandas import DataFrame

from drl_trading_framework.common.config.feature_config_factory import (
    FeatureConfigFactory,
)
from drl_trading_framework.common.config.utils import parse_all_parameters
from drl_trading_framework.preprocess.feature.collection.base_feature import BaseFeature
from drl_trading_framework.preprocess.feature.feature_class_factory import (
    FeatureClassFactory,
)
from drl_trading_framework.preprocess.metrics.technical_metrics_service import (
    TechnicalMetricsServiceInterface,
)


class RsiConfig(BaseParameterSetConfig):
    type: Literal["rsi"]
    length: int


class RsiFeature(BaseFeature):

    def __init__(
        self,
        source: DataFrame,
        config: BaseParameterSetConfig,
        postfix: str = "",
        metrics_service: Optional[TechnicalMetricsServiceInterface] = None,
    ) -> None:
        super().__init__(source, config, postfix, metrics_service)
        self.config: RsiConfig = self.config

    def compute(self) -> DataFrame:
        # Get source DataFrame with ensured DatetimeIndex using the base class method
        source_df = self._prepare_source_df()

        # Create a DataFrame with the same index as the source
        rsi_values = ta.rsi(source_df["Close"], length=self.config.length)

        # Create result DataFrame with both Time column and feature values
        df = DataFrame(index=source_df.index)
        df[f"rsi_{self.config.length}{self.postfix}"] = rsi_values

        return df

    def get_sub_features_names(self) -> list[str]:
        return [f"rsi_{self.config.length}{self.postfix}"]


@pytest.fixture
def feature_config_factory():
    """Create a fresh feature config factory instance for testing."""
    factory = FeatureConfigFactory()
    factory.discover_config_classes(package_name="tests")
    return factory


@pytest.fixture
def feature_class_registry():
    reg = FeatureClassFactory()
    reg.discover_feature_classes(package_name="tests")
    return reg


@pytest.fixture(scope="session")
def mocked_config():
    """Load test configuration from the test resources directory."""
    config_path = os.path.join(
        os.path.dirname(__file__), "resources/applicationConfig-test.json"
    )
    return ConfigLoader.get_config(config_path)


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


@pytest.fixture
def mocked_container(feature_config_factory, feature_class_registry):
    """Create a mocked dependency injection container using the injector library.

    This fixture provides a configured injector instance with test dependencies
    for integration testing, replacing the legacy dependency-injector approach.
    """
    # Import necessary types for the injector setup
    from injector import Injector, Module, provider, singleton

    from drl_trading_framework.common.config.feature_config_factory import (
        FeatureConfigFactoryInterface,
    )
    from drl_trading_framework.preprocess.feature.feature_class_factory import (
        FeatureClassFactoryInterface,
    )

    # Use the test config path
    test_config_path = os.path.join(
        os.path.dirname(__file__), "resources/applicationConfig-test.json"
    )

    class TestModule(Module):
        """Test module that provides factory instances and config path."""

        @provider
        @singleton
        def provide_config_path(self) -> str:
            return test_config_path

        @provider
        @singleton
        def provide_feature_config_factory(self) -> FeatureConfigFactoryInterface:
            return feature_config_factory

        @provider
        @singleton
        def provide_feature_class_factory(self) -> FeatureClassFactoryInterface:
            return feature_class_registry

    # Import and create the application module
    from drl_trading_framework.common.di.application_container import DomainModule

    # Create injector with both modules - TestModule provides the factories,
    # ApplicationModule provides the rest of the configuration
    app_module = DomainModule(config_path=test_config_path)
    test_module = TestModule()
    injector = Injector([test_module, app_module])

    return injector


@pytest.fixture(scope="session")
def mocked_feature_store(request, mocked_config):
    """Initialize the Feast feature store for testing.

    Sets up a clean feature store for testing and ensures proper cleanup
    after the test session completes.

    Args:
        request: pytest request object
        mocked_config: The test configuration fixture

    Returns:
        FeatureStore: Initialized feature store instance
    """
    # Get paths from config
    repo_path = mocked_config.feature_store_config.repo_path
    store_path = mocked_config.feature_store_config.offline_store_path

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if not os.path.isabs(repo_path):
        abs_repo_path = os.path.join(project_root, repo_path)
    else:
        abs_repo_path = repo_path

    if not os.path.isabs(store_path):
        abs_store_path = os.path.join(project_root, store_path)
    else:
        abs_store_path = store_path

    # Clean up any existing feature store data before starting
    _clean_feature_store(abs_repo_path, abs_store_path)

    # Create data directory within repo path
    data_dir = os.path.join(abs_repo_path, "data")
    Path(data_dir).mkdir(exist_ok=True)

    # Initialize the repository
    try:
        subprocess.run(
            ["feast", "apply"], cwd=abs_repo_path, check=True, capture_output=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error initializing feast: {e.stderr.decode()}")
        raise

    # Return the FeatureStore instance
    feature_store = FeatureStore(repo_path=abs_repo_path)

    # Clean up function for after the session
    def cleanup():
        """Remove feature store data after test session completes."""
        _clean_feature_store(abs_repo_path, abs_store_path)
        print("Feature store cleaned up successfully after test session.")

    request.addfinalizer(cleanup)
    return feature_store


@pytest.fixture
def reset_registry_in_container(mocked_container):
    """
    Reset registries in an existing container instance.

    This fixture is useful for tests that need a mocked_container
    but also require clean registry state.

    Args:
        mocked_container: The injector instance to update

    Returns:
        The same injector instance with reset registries
    """    # Import needed classes and interfaces
    from drl_trading_common.config.application_config import ApplicationConfig

    from drl_trading_framework.common.config.feature_config_factory import (
        FeatureConfigFactoryInterface,
    )
    from drl_trading_framework.preprocess.feature.feature_class_factory import (
        FeatureClassFactoryInterface,
    )

    feature_class_registry = mocked_container.get(FeatureClassFactoryInterface)
    feature_class_registry.reset()

    # Reset the feature config factory
    feature_config_factory = mocked_container.get(FeatureConfigFactoryInterface)
    feature_config_factory.clear()
    feature_config_factory.discover_config_classes()# Re-parse feature configurations
    app_config = mocked_container.get(ApplicationConfig)
    parse_all_parameters(app_config.features_config, feature_config_factory)

    return mocked_container


def _clean_feature_store(repo_path: str, store_path: str) -> None:
    """Helper function to clean up feature store data.

    Args:
        repo_path: Path to the feature store repository
        store_path: Path to the feature store data directory
    """
    # Clean the registry DB
    registry_db = os.path.join(repo_path, "data", "registry.db")
    if os.path.exists(registry_db):
        os.remove(registry_db)
        print(f"Removed feature store registry: {registry_db}")

    # Clean the feature store data directory
    if os.path.exists(store_path):
        shutil.rmtree(store_path)
        print(f"Removed feature store data directory: {store_path}")
        # Recreate the empty directory
        os.makedirs(store_path, exist_ok=True)
