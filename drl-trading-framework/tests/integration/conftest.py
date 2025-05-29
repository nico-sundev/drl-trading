import os
import shutil
import subprocess
from pathlib import Path
from typing import Literal, Optional
from unittest.mock import MagicMock

import pandas_ta as ta
import pytest
from drl_trading_common import BaseParameterSetConfig
from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.config.application_config import ApplicationConfig
from drl_trading_common.config.config_loader import ConfigLoader
from drl_trading_common.interfaces.feature.feature_class_registry_interface import (
    FeatureClassRegistryInterface,
)
from drl_trading_common.interfaces.feature.feature_config_registry_interface import (
    FeatureConfigRegistryInterface,
)
from feast import FeatureStore
from injector import Injector
from pandas import DataFrame

from drl_trading_framework.preprocess.feature.feature_factory import (
    FeatureFactory,
    FeatureFactoryInterface,
)
from drl_trading_framework.preprocess.metrics.technical_metrics_service import (
    TechnicalMetricsServiceInterface,
)


@pytest.fixture(scope="session")
def mocked_config() -> ApplicationConfig:
    """Load test configuration from the test resources directory."""
    config_path = os.path.join(
        os.path.dirname(__file__), "../resources/applicationConfig-test.json"
    )
    return ConfigLoader.get_config(config_path)


@pytest.fixture(scope="session")
def mocked_feature_store(request, mocked_config: ApplicationConfig) -> FeatureStore:
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

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
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


@pytest.fixture(scope="session")
def feature_config_registry():
    """Create a mock feature config registry for testing."""
    mock = MagicMock(spec=FeatureConfigRegistryInterface)

    # Define mapping of feature types to config classes
    config_mapping = {
        "rsi": RsiConfig,
    }

    def get_config_class(feature_type: str):
        return config_mapping.get(feature_type.lower())

    mock.get_config_class.side_effect = get_config_class
    mock.reset.side_effect = lambda: None
    return mock

@pytest.fixture(scope="session")
def feature_class_registry():
    """Create a mock feature class registry that returns RsiFeature for 'rsi' type."""
    mock = MagicMock(spec=FeatureClassRegistryInterface)

    # Define mapping of feature types to feature classes
    feature_mapping = {
        "rsi": RsiFeature
    }

    def get_feature_class(feature_type: str):
        if feature_type in feature_mapping:
            return feature_mapping[feature_type]
        raise ValueError(f"Unknown feature type: {feature_type}")

    mock.get_feature_class.side_effect = get_feature_class
    return mock

@pytest.fixture(scope="session")
def feature_factory(feature_class_registry, feature_config_registry):
    """Create a feature factory instance for testing."""
    return FeatureFactory(feature_class_registry, feature_config_registry)

@pytest.fixture(scope="session")
def mocked_container(feature_factory, mocked_feature_store):
    """Create a mocked dependency injection container using the injector library.

    This fixture provides a configured injector instance with test dependencies
    for integration testing, replacing the legacy dependency-injector approach.
    """
    # Import necessary types for the injector setup

    # Use the test config path
    test_config_path = os.path.join(
        os.path.dirname(__file__), "../resources/applicationConfig-test.json"
    )    # Import and create the application module
    from drl_trading_framework.common.di.domain_module import DomainModule

    # Create injector with the domain module
    app_module = DomainModule(config_path=test_config_path)
    injector = Injector([app_module])

    # Override the feature factory with our test fixture
    injector.binder.bind(FeatureFactoryInterface, to=feature_factory)

    # Override the feature store with our test fixture
    injector.binder.bind(FeatureStore, to=mocked_feature_store)

    return injector
