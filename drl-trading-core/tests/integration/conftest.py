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
from drl_trading_common.interfaces.technical_indicator_service_interface import (
    TechnicalIndicatorFacadeInterface,
)
from feast import FeatureStore
from injector import Injector
from pandas import DataFrame

from drl_trading_core.preprocess.feature.feature_factory import (
    FeatureFactory,
    FeatureFactoryInterface,
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
    """Mock RSI feature implementation for testing."""

    def __init__(
        self,
        source: DataFrame,
        config: BaseParameterSetConfig,
        indicator_service: TechnicalIndicatorFacadeInterface,
        postfix: str = "",
    ) -> None:
        super().__init__(source, config, indicator_service, postfix)
        self.config: RsiConfig = self.config
        self.feature_name = f"rsi_{self.config.length}{self.postfix}"
        # Mock the indicator service registration for testing

    def add(self, df: DataFrame) -> None:
        """Add data to the feature (mock implementation)."""
        pass

    def compute_latest(self) -> Optional[DataFrame]:
        """Compute latest RSI value (mock implementation)."""
        if hasattr(self, 'df_source') and not self.df_source.empty:
            latest_rsi = ta.rsi(close=self.df_source["Close"], length=self.config.length).iloc[-1:]
            result_df = DataFrame(index=self.df_source.index[-1:])
            result_df[f"rsi_{self.config.length}{self.postfix}"] = latest_rsi
            return result_df
        return None

    def compute_all(self) -> Optional[DataFrame]:
        """Compute all RSI values (mock implementation)."""
        if hasattr(self, 'df_source') and not self.df_source.empty:
            rsi_values = ta.rsi(close=self.df_source["Close"], length=self.config.length)
            result_df = DataFrame(index=self.df_source.index)
            result_df[f"rsi_{self.config.length}{self.postfix}"] = rsi_values
            return result_df
        return None

    def get_sub_features_names(self) -> list[str]:
        """Get sub-feature names."""
        return [f"rsi_{self.config.length}{self.postfix}"]

    def get_feature_name(self) -> str:
        """Get feature name."""
        return "rsi"


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
    from drl_trading_core.common.di.core_module import CoreModule

    # Create injector with the domain module
    app_module = CoreModule(config_path=test_config_path)
    injector = Injector([app_module])

    # Override the feature factory with our test fixture
    injector.binder.bind(FeatureFactoryInterface, to=feature_factory)

    # Override the feature store with our test fixture
    injector.binder.bind(FeatureStore, to=mocked_feature_store)

    return injector
