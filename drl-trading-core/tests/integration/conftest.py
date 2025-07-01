import json
import os
import shutil
import tempfile
from datetime import datetime
from typing import Generator, Optional

import pandas as pd
import pytest
from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_common.config.application_config import ApplicationConfig
from drl_trading_common.config.config_loader import ConfigLoader
from drl_trading_common.decorator.feature_role_decorator import feature_role
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.interface.feature.feature_factory_interface import (
    IFeatureFactory,
)
from drl_trading_common.interface.indicator.technical_indicator_facade_interface import (
    ITechnicalIndicatorFacade,
)
from drl_trading_common.model.dataset_identifier import DatasetIdentifier
from drl_trading_common.model.feature_config_version_info import (
    FeatureConfigVersionInfo,
)
from feast import FeatureStore
from injector import Injector
from pandas import DataFrame


# Test Feature Implementations for Core Integration Testing
class MockTechnicalIndicatorFacade(ITechnicalIndicatorFacade):
    """Mock technical indicator facade that returns controlled test data for integration testing."""

    def __init__(self) -> None:
        self._indicators: dict[str, DataFrame] = {}

    def register_instance(self, name: str, indicator_type, **params) -> None:
        """Register a mock indicator that returns predictable test data."""
        # Create predictable test data based on indicator type
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1H")

        if "rsi" in name.lower():
            # Generate RSI-like values between 30-70
            values = [50.0 + (i % 20) for i in range(100)]
            self._indicators[name] = pd.DataFrame({
                "event_timestamp": dates,
                name: values
            })
        elif "close" in name.lower():
            # Generate price-like values
            values = [1.1000 + (i % 50) * 0.0001 for i in range(100)]
            self._indicators[name] = pd.DataFrame({
                "event_timestamp": dates,
                name: values
            })
        else:
            # Default values
            values = [float(i) for i in range(100)]
            self._indicators[name] = pd.DataFrame({
                "event_timestamp": dates,
                name: values
            })

    def add(self, name: str, value: DataFrame) -> None:
        """Mock incremental computation - not needed for current tests."""
        pass

    def get_all(self, name: str) -> Optional[DataFrame]:
        """Return all mock indicator data."""
        return self._indicators.get(name)

    def get_latest(self, name: str) -> Optional[DataFrame]:
        """Return latest mock indicator data."""
        data = self._indicators.get(name)
        return data.tail(1) if data is not None else None


class TestRsiConfig(BaseParameterSetConfig):
    """Test configuration for RSI feature that follows the existing pattern."""

    type: str = "rsi"
    enabled: bool = True
    period: int = 14

    def hash_id(self) -> str:
        return f"rsi_{self.period}"


class TestClosePriceConfig(BaseParameterSetConfig):
    """Test configuration for close price feature that follows the existing pattern."""

    type: str = "close_price"
    enabled: bool = True
    lookback: int = 1

    def hash_id(self) -> str:
        return f"close_{self.lookback}"


@feature_role(FeatureRoleEnum.OBSERVATION_SPACE)
class TestRsiFeature(BaseFeature):
    """Test RSI feature implementation adapted from existing MockFeature pattern."""

    def __init__(self, config: TestRsiConfig, dataset_id: DatasetIdentifier, indicator_service: MockTechnicalIndicatorFacade, postfix: str = ""):
        super().__init__(config, dataset_id, indicator_service, postfix)
        self._feature_name = "rsi"
        # Register the indicator when feature is created
        self.indicator_service.register_instance(f"rsi_{config.period}", "rsi", period=config.period)

    def get_feature_name(self) -> str:
        return self._feature_name

    def get_sub_features_names(self) -> list[str]:
        return [f"rsi_{self.config.period}"]

    def compute_all(self) -> Optional[DataFrame]:
        """Compute RSI using the mock indicator service."""
        indicator_name = f"rsi_{self.config.period}"
        indicator_data = self.indicator_service.get_all(indicator_name)

        if indicator_data is None:
            return None

        # Add the required symbol column for Feast compatibility
        result = indicator_data.copy()
        result[self.dataset_id.symbol] = self.dataset_id.symbol
        return result

    def add(self, df: DataFrame) -> None:
        """Mock incremental computation - not implemented for current testing."""
        pass

    def compute_latest(self) -> Optional[DataFrame]:
        """Get latest RSI values."""
        indicator_name = f"rsi_{self.config.period}"
        indicator_data = self.indicator_service.get_latest(indicator_name)

        if indicator_data is None:
            return None

        result = indicator_data.copy()
        result[self.dataset_id.symbol] = self.dataset_id.symbol
        return result


@feature_role(FeatureRoleEnum.REWARD_ENGINEERING)
class TestClosePriceFeature(BaseFeature):
    """Test close price feature implementation adapted from existing MockFeature pattern."""

    def __init__(self, config: TestClosePriceConfig, dataset_id: DatasetIdentifier, indicator_service: MockTechnicalIndicatorFacade, postfix: str = ""):
        super().__init__(config, dataset_id, indicator_service, postfix)
        self._feature_name = "close_price"
        # Register the indicator when feature is created
        self.indicator_service.register_instance(f"close_{config.lookback}", "close", lookback=config.lookback)

    def get_feature_name(self) -> str:
        return self._feature_name

    def get_sub_features_names(self) -> list[str]:
        return [f"close_{self.config.lookback}"]

    def compute_all(self) -> Optional[DataFrame]:
        """Compute close prices using the mock indicator service."""
        indicator_name = f"close_{self.config.lookback}"
        indicator_data = self.indicator_service.get_all(indicator_name)

        if indicator_data is None:
            return None

        # Add the required symbol column for Feast compatibility
        result = indicator_data.copy()
        result[self.dataset_id.symbol] = self.dataset_id.symbol
        return result

    def add(self, df: DataFrame) -> None:
        """Mock incremental computation - not implemented for current testing."""
        pass

    def compute_latest(self) -> Optional[DataFrame]:
        """Get latest close price values."""
        indicator_name = f"close_{self.config.lookback}"
        indicator_data = self.indicator_service.get_latest(indicator_name)

        if indicator_data is None:
            return None

        result = indicator_data.copy()
        result[self.dataset_id.symbol] = self.dataset_id.symbol
        return result


class TestFeatureFactory(IFeatureFactory):
    """Test feature factory adapted from existing patterns that creates minimal features for integration testing."""

    def __init__(self):
        self.indicator_service = MockTechnicalIndicatorFacade()

    def create_feature(
        self,
        feature_name: str,
        dataset_id: DatasetIdentifier,
        config: BaseParameterSetConfig,
        postfix: str = ""
    ) -> Optional[BaseFeature]:
        """Create test feature instances using the same pattern as real factories."""
        if feature_name == "rsi" and isinstance(config, TestRsiConfig):
            return TestRsiFeature(config, dataset_id, self.indicator_service, postfix)
        elif feature_name == "close_price" and isinstance(config, TestClosePriceConfig):
            return TestClosePriceFeature(config, dataset_id, self.indicator_service, postfix)
        return None

    def create_config_instance(
        self, feature_name: str, config_data: dict
    ) -> Optional[BaseParameterSetConfig]:
        """Create test configuration instances following existing patterns."""
        if feature_name == "rsi":
            period = config_data.get("period", 14)
            return TestRsiConfig(period=period)
        elif feature_name == "close_price":
            lookback = config_data.get("lookback", 1)
            return TestClosePriceConfig(lookback=lookback)
        return None


@pytest.fixture(scope="session")
def test_feature_factory() -> TestFeatureFactory:
    """Provide a test feature factory for integration testing."""
    return TestFeatureFactory()


@pytest.fixture(scope="session")
def temp_feast_repo() -> Generator[str, None, None]:
    """Create a temporary directory for Feast repository during testing."""
    temp_dir = tempfile.mkdtemp(prefix="feast_integration_test_")
    yield temp_dir
    # Cleanup after all tests are done
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def config_fixture(temp_feast_repo: str) -> ApplicationConfig:
    """Load test configuration and override Feast paths with temp directory."""
    config_path = os.path.join(
        os.path.dirname(__file__), "../resources/applicationConfig-test.json"
    )
    config = ConfigLoader.get_config(ApplicationConfig, path=config_path)

    # Override feature store configuration to use temp directory
    config.feature_store_config.repo_path = temp_feast_repo
    config.feature_store_config.enabled = True

    return config


@pytest.fixture(scope="session")
def feature_store_fixture(temp_feast_repo: str) -> FeatureStore:
    """Initialize the Feast feature store for testing with file-based configuration.

    Uses file-based configuration with relative paths to avoid Windows
    drive letter issues in Feast's URI parsing.

    Args:
        temp_feast_repo: Path to the temporary Feast repository

    Returns:
        FeatureStore: Initialized feature store instance with real backend
    """
    # Create feature_store.yaml file with relative paths to avoid Windows issues
    feature_store_yaml = """
project: test_trading_project
registry: registry.db
provider: local
online_store:
    type: sqlite
    path: online_store.db
offline_store:
    type: file
entity_key_serialization_version: 2
"""

    config_path = os.path.join(temp_feast_repo, "feature_store.yaml")
    with open(config_path, "w") as f:
        f.write(feature_store_yaml)

    # Initialize the feature store with the repository path
    feature_store = FeatureStore(repo_path=temp_feast_repo)
    return feature_store


@pytest.fixture(scope="function")
def clean_feature_store(feature_store_fixture: FeatureStore, temp_feast_repo: str) -> Generator[FeatureStore, None, None]:
    """Provide a clean feature store for each test function.

    This fixture ensures each test starts with a clean slate by clearing
    the feature store data before and after each test.
    """
    # Clean before test
    _clear_feature_store_data(temp_feast_repo)

    yield feature_store_fixture

    # Clean after test
    _clear_feature_store_data(temp_feast_repo)


def _clear_feature_store_data(temp_repo_path: str) -> None:
    """Clear all data from the feature store by removing database files.

    Since we're using a temporary directory with relative paths in Feast config,
    we need to remove the database files from the repository directory.

    Args:
        temp_repo_path: Path to the temporary Feast repository
    """
    try:
        # Remove registry database (relative to repo path)
        registry_db = os.path.join(temp_repo_path, "registry.db")
        if os.path.exists(registry_db):
            os.remove(registry_db)

        # Remove online store database (relative to repo path)
        online_store_db = os.path.join(temp_repo_path, "online_store.db")
        if os.path.exists(online_store_db):
            os.remove(online_store_db)

        # Remove any data directories
        data_dir = os.path.join(temp_repo_path, "data")
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)

        # Remove feature_store.yaml to force fresh config on next test
        config_file = os.path.join(temp_repo_path, "feature_store.yaml")
        if os.path.exists(config_file):
            # Don't remove - we need it for the FeatureStore to work
            pass

    except Exception as e:
        # Log the error but don't fail the test
        print(f"Warning: Could not clear feature store data: {e}")


@pytest.fixture(scope="function")
def feature_version_info_fixture() -> FeatureConfigVersionInfo:
    """Create feature version info for integration testing."""
    return FeatureConfigVersionInfo(
        semver="1.0.0",
        hash="integration_test_hash_123",
        created_at=datetime.fromisoformat("2024-01-01T00:00:00"),
        feature_definitions=[
            {"name": "rsi_14", "enabled": True, "role": "observation_space"},
            {"name": "close_price_1", "enabled": True, "role": "reward_engineering"}
        ]
    )


@pytest.fixture(scope="function")
def real_feast_container(test_feature_factory: TestFeatureFactory, config_fixture: ApplicationConfig, temp_feast_repo: str) -> Injector:
    """Create a dependency injection container with REAL Feast integration.

    This fixture provides a configured injector instance with real services
    for true integration testing. Uses minimal test features that don't depend
    on the strategy module, maintaining proper dependency boundaries.

    Args:
        test_feature_factory: Test feature factory with RSI and close price features
        config_fixture: Test configuration with temp Feast repository
        temp_feast_repo: Path to the temporary Feast repository

    Returns:
        Injector: Configured DI container with real services and test features
    """
    from drl_trading_core.common.di.core_module import CoreModule

    # Write config to a temporary JSON file so CoreModule can load it
    # Use a temp file that won't be auto-deleted until we're done
    temp_fd, config_path = tempfile.mkstemp(suffix='.json', text=True)

    with os.fdopen(temp_fd, 'w') as config_file:
        # Load the existing test config and modify it for the temp Feast repo
        test_config_path = os.path.join(
            os.path.dirname(__file__), "../resources/applicationConfig-test.json"
        )

        with open(test_config_path, 'r') as test_config_file:
            config_dict = json.load(test_config_file)

        # Override the feature store configuration to use temp directory
        config_dict["featureStoreConfig"] = {
            "enabled": True,
            "repo_path": temp_feast_repo,
            "entity_name": config_fixture.feature_store_config.entity_name,
            "ttl_days": config_fixture.feature_store_config.ttl_days,
            "online_enabled": False,
            "service_name": config_fixture.feature_store_config.service_name,
            "service_version": config_fixture.feature_store_config.service_version
        }

        json.dump(config_dict, config_file, indent=2)

    # Create injector with the real Feast configuration and CORE test modules only
    app_module = CoreModule(config_path=config_path)
    injector = Injector([app_module])

    # Override the feature factory with our test implementation
    # This maintains the dependency boundary - core tests don't depend on strategy
    injector.binder.bind(IFeatureFactory, to=test_feature_factory)

    # Store the config path on the injector so we can clean it up later
    # We need to use a workaround for type checking since Injector doesn't have this attribute
    injector._test_config_path = config_path

    return injector


@pytest.fixture(scope="function")
def integration_container(real_feast_container: Injector, clean_feature_store: FeatureStore) -> Generator[Injector, None, None]:
    """Provide container with clean feature store for each test.

    This ensures each test gets a fresh, clean feature store state while
    using real Feast infrastructure throughout the integration test.

    Args:
        real_feast_container: DI container with real services
        clean_feature_store: Clean feature store instance

    Returns:
        Injector: Ready-to-use DI container for integration testing
    """
    yield real_feast_container

    # Cleanup the temporary config file if it exists
    config_path = getattr(real_feast_container, '_test_config_path', None)
    if config_path:
        try:
            os.unlink(config_path)
        except OSError:
            pass  # File might already be deleted
