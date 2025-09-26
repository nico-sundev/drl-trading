import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from typing import Generator, Optional

import boto3
import pandas as pd
import pytest
from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_common.config.application_config import ApplicationConfig
from drl_trading_common.config.config_loader import ConfigLoader
from drl_trading_common.config.feature_config import (
    FeatureStoreConfig,
    LocalRepoConfig,
    S3RepoConfig,
)
from drl_trading_common.decorator.feature_role_decorator import feature_role
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.enum.offline_repo_strategy_enum import OfflineRepoStrategyEnum
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
from injector import Injector
from pandas import DataFrame
from testcontainers.minio import MinioContainer


@pytest.fixture(scope="session", autouse=True)
def register_cleanup_on_exit():
    """Register cleanup to run when Python exits - much simpler and more reliable.

    Since we use unique database files per test, we don't need complex session cleanup.
    The atexit handler runs when all file handles are guaranteed to be closed.
    """
    import atexit

    test_config_dir = os.path.join(os.path.dirname(__file__), "../resources/test")
    data_dir = os.path.join(test_config_dir, "data")

    # Register cleanup to run when Python process exits
    atexit.register(_cleanup_on_exit, data_dir)

    yield  # Let tests run


def _cleanup_on_exit(data_dir: str) -> None:
    """Clean up test data when Python exits - all file handles are closed by then."""
    try:
        if os.path.exists(data_dir):
            import shutil
            shutil.rmtree(data_dir, ignore_errors=True)
            print(f"DEBUG: Exit cleanup completed for: {data_dir}")
    except Exception as e:
        print(f"DEBUG: Exit cleanup best effort: {e}")  # Log but don't fail


@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Configure logging for integration tests with debug level."""
    # Set up logging configuration for integration tests
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # Override any existing logging configuration
    )

    # Set specific loggers to DEBUG level
    loggers = ["drl_trading_core", "drl_trading_common", "feast", "root"]

    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

    # Also set the root logger
    logging.getLogger().setLevel(logging.DEBUG)

    print("DEBUG: Integration test logging configured to DEBUG level")


# Test Feature Implementations for Core Integration Testing
class MockTechnicalIndicatorFacade(ITechnicalIndicatorFacade):
    """Mock technical indicator facade that returns controlled test data for integration testing."""

    def __init__(self) -> None:
        self._indicators: dict[str, DataFrame] = {}

    def register_instance(self, name: str, indicator_type, **params) -> None:
        """Register a mock indicator that returns predictable test data."""
        # Create predictable test data based on indicator type with UTC timezone
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1H", tz="UTC")

        if "rsi" in name.lower():
            # Generate RSI-like values between 30-70
            values = [50.0 + (i % 20) for i in range(100)]
            self._indicators[name] = pd.DataFrame(
                {"event_timestamp": dates, name: values}
            )
        elif "close_price" in name.lower():
            # Generate price-like values
            values = [1.1000 + (i % 50) * 0.0001 for i in range(100)]
            self._indicators[name] = pd.DataFrame(
                {"event_timestamp": dates, name: values}
            )
        elif name == "reward":
            # Generate reward-like values between -0.1 and 0.1
            values = [0.01 * (i % 20 - 10) for i in range(100)]
            self._indicators[name] = pd.DataFrame(
                {"event_timestamp": dates, "reward": values}
            )
        elif name == "cumulative_return":
            # Generate cumulative return values
            values = [0.001 * i for i in range(100)]
            self._indicators[name] = pd.DataFrame(
                {"event_timestamp": dates, "cumulative_return": values}
            )
        else:
            # Default values
            values = [float(i) for i in range(100)]
            self._indicators[name] = pd.DataFrame(
                {"event_timestamp": dates, name: values}
            )

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
        return "A1b2c3"


@feature_role(FeatureRoleEnum.OBSERVATION_SPACE)
class TestRsiFeature(BaseFeature):
    """Test RSI feature implementation adapted from existing MockFeature pattern."""

    def __init__(
        self,
        dataset_id: DatasetIdentifier,
        indicator_service: MockTechnicalIndicatorFacade,
        config: TestRsiConfig,
        postfix: str = "",
    ):
        super().__init__(dataset_id, indicator_service, config, postfix)
        self._feature_name = "rsi"
        # Register the indicator when feature is created
        self.indicator_service.register_instance(
            f"rsi_{config.period}", "rsi", period=config.period
        )

    def get_feature_name(self) -> str:
        return self._feature_name

    def get_sub_features_names(self) -> list[str]:
        return []

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

    def update(self, df: DataFrame) -> None:
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

    def get_config_to_string(self) -> str:
        return f"{self.config.period}"


@feature_role(FeatureRoleEnum.OBSERVATION_SPACE)
class TestClosePriceFeature(BaseFeature):
    """Test close price feature implementation adapted from existing MockFeature pattern."""

    def __init__(
        self,
        dataset_id: DatasetIdentifier,
        indicator_service: MockTechnicalIndicatorFacade,
        config: Optional[BaseParameterSetConfig] = None,
        postfix: str = "",
    ):
        super().__init__(dataset_id, indicator_service, config, postfix)
        self._feature_name = "close_price"

    def get_feature_name(self) -> str:
        return self._feature_name

    def get_sub_features_names(self) -> list[str]:
        return []

    def compute_all(self) -> Optional[DataFrame]:
        """Compute close prices using the mock indicator service."""
        indicator_name = "close_price"
        indicator_data = self.indicator_service.get_all(indicator_name)

        if indicator_data is None:
            return None

        # Add the required symbol column for Feast compatibility
        result = indicator_data.copy()
        result[self.dataset_id.symbol] = self.dataset_id.symbol
        return result

    def update(self, df: DataFrame) -> None:
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

    def get_config_to_string(self) -> Optional[str]:
        return None


@feature_role(FeatureRoleEnum.REWARD_ENGINEERING)
class TestRewardFeature(BaseFeature):
    """Test reward feature implementation that produces reward and cumulative_return."""

    def __init__(
        self,
        dataset_id: DatasetIdentifier,
        indicator_service: MockTechnicalIndicatorFacade,
        config: Optional[BaseParameterSetConfig] = None,
        postfix: str = "",
    ):
        super().__init__(dataset_id, indicator_service, config, postfix)
        self._feature_name = "reward"
        # Register the indicators when feature is created
        self.indicator_service.register_instance(
            "reward", "reward"
        )
        self.indicator_service.register_instance(
            "cumulative_return", "cumulative_return"
        )

    def get_feature_name(self) -> str:
        return self._feature_name

    def get_sub_features_names(self) -> list[str]:
        return ["reward", "cumulative_return"]

    def compute_all(self) -> Optional[DataFrame]:
        """Compute reward features using the mock indicator service."""
        # Get both reward and cumulative return data
        reward_data = self.indicator_service.get_all("reward")
        cumulative_data = self.indicator_service.get_all("cumulative_return")

        if reward_data is None or cumulative_data is None:
            return None

        # Combine both into a single DataFrame
        result = reward_data.copy()
        result["cumulative_return"] = cumulative_data["cumulative_return"]

        # Add the required symbol column for Feast compatibility
        result[self.dataset_id.symbol] = self.dataset_id.symbol
        return result

    def update(self, df: DataFrame) -> None:
        """Mock incremental computation - not implemented for current testing."""
        pass

    def compute_latest(self) -> Optional[DataFrame]:
        """Get latest reward feature values."""
        reward_data = self.indicator_service.get_latest("reward")
        cumulative_data = self.indicator_service.get_latest("cumulative_return")

        if reward_data is None or cumulative_data is None:
            return None

        result = reward_data.copy()
        result["cumulative_return"] = cumulative_data["cumulative_return"]
        result[self.dataset_id.symbol] = self.dataset_id.symbol
        return result

    def get_config_to_string(self) -> Optional[str]:
        return None


class TestFeatureFactory(IFeatureFactory):
    """Test feature factory adapted from existing patterns that creates minimal features for integration testing."""

    def __init__(self):
        self.indicator_service = MockTechnicalIndicatorFacade()

    def create_feature(
        self,
        feature_name: str,
        dataset_id: DatasetIdentifier,
        config: BaseParameterSetConfig,
        postfix: str = "",
    ) -> Optional[BaseFeature]:
        """Create test feature instances using the same pattern as real factories."""
        if feature_name == "rsi" and isinstance(config, TestRsiConfig):
            return TestRsiFeature(dataset_id, self.indicator_service, config, postfix)
        elif feature_name == "close_price":
            # Close price feature can handle None config
            return TestClosePriceFeature(
                dataset_id, self.indicator_service, config, postfix
            )
        elif feature_name == "reward":
            return TestRewardFeature(
                dataset_id, self.indicator_service, config, postfix
            )
        return None

    def create_config_instance(
        self, feature_name: str, config_data: dict
    ) -> Optional[BaseParameterSetConfig]:
        """Create test configuration instances following existing patterns."""
        if feature_name == "rsi":
            period = config_data.get("period", 14)
            return TestRsiConfig(period=period)
        elif feature_name == "close_price":
            # Close price feature can handle None config
            return None
        elif feature_name == "reward":
            return None
        return None


@pytest.fixture(scope="session")
def test_feature_factory() -> TestFeatureFactory:
    """Provide a test feature factory for integration testing."""
    return TestFeatureFactory()


@pytest.fixture
def test_rsi_config() -> TestRsiConfig:
    """Fixture providing test RSI configuration."""
    return TestRsiConfig()

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

    # Override feature store configuration to use static test config with environment variables
    from drl_trading_common.config.feature_config import LocalRepoConfig
    from drl_trading_common.enum.offline_repo_strategy_enum import (
        OfflineRepoStrategyEnum,
    )

    # Set the config_directory to point to our static test configuration
    test_config_dir = os.path.join(os.path.dirname(__file__), "../resources")

    # Create new local repo config pointing to test config directory
    # This allows Feast to use relative paths like "data/registry.db"
    local_repo_config = LocalRepoConfig(repo_path=test_config_dir)

    # Set the strategy and config
    config.feature_store_config.offline_repo_strategy = OfflineRepoStrategyEnum.LOCAL
    config.feature_store_config.local_repo_config = local_repo_config
    config.feature_store_config.config_directory = str(test_config_dir)  # Points to tests/resources
    config.feature_store_config.cache_enabled = True

    return config


@pytest.fixture(scope="function")
def clean_integration_environment() -> Generator[None, None, None]:
    """Clean integration test environment with proper isolation.

    Uses unique database paths per test to avoid file locking issues.
    This is the proper way to handle stateful resources in testing.
    """
    import uuid

    # Set STAGE environment variable for consistent testing
    os.environ.setdefault("STAGE", "test")

    # Use unique database paths per test to avoid file locking completely
    # This is the standard approach for database testing
    test_uuid = uuid.uuid4().hex[:8]
    os.environ["FEAST_REGISTRY_PATH"] = f"data/registry_{test_uuid}.db"
    os.environ["FEAST_ONLINE_STORE_PATH"] = f"data/online_store_{test_uuid}.db"

    yield

    # Clean up environment variables (files will be cleaned up by session fixture)
    for env_var in ["FEAST_REGISTRY_PATH", "FEAST_ONLINE_STORE_PATH", "FEAST_OFFLINE_STORE_PATH"]:
        os.environ.pop(env_var, None)


def _initialize_feast_repository(repo_path: str) -> None:
    """Initialize a Feast repository with staged configuration.

    For integration tests, this simply ensures the STAGE environment variable
    is set to 'test' to use the existing test configuration.

    Args:
        repo_path: Path to the directory where the Feast repository should be created
    """
    # Ensure the directory exists
    os.makedirs(repo_path, exist_ok=True)

    # Set STAGE for test if not already set
    os.environ.setdefault("STAGE", "test")

    # No need to copy configs - the test configuration already exists at
    # tests/resources/test/feature_store.yaml and will be used by FeatureStoreWrapper


@pytest.fixture(scope="function")
def feature_version_info_fixture() -> FeatureConfigVersionInfo:
    """Create feature version info for integration testing."""
    return FeatureConfigVersionInfo(
        semver="1.0.0",
        hash="integration_test_hash_123",
        created_at=datetime.fromisoformat("2024-01-01T00:00:00"),
        feature_definitions=[
            {"name": "rsi_14", "enabled": True, "role": "observation_space"},
            {"name": "close_1", "enabled": True, "role": "observation_space"},
            {"name": "reward", "enabled": True, "role": "reward_engineering"},
            {
                "name": "cumulative_return",
                "enabled": True,
                "role": "reward_engineering",
            },
        ],
    )


@pytest.fixture(scope="function")
def real_feast_container(
    test_feature_factory: TestFeatureFactory,
    config_fixture: ApplicationConfig,
    temp_feast_repo: str,
) -> Injector:
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

    # Ensure STAGE is set for consistent testing
    os.environ.setdefault("STAGE", "test")

    # Write config to a temporary JSON file so CoreModule can load it
    # Use a temp file that won't be auto-deleted until we're done
    temp_fd, config_path = tempfile.mkstemp(suffix=".json", text=True)

    with os.fdopen(temp_fd, "w") as config_file:
        # Load the existing test config and modify it for the temp Feast repo
        test_config_path = os.path.join(
            os.path.dirname(__file__), "../resources/applicationConfig-test.json"
        )

        with open(test_config_path, "r") as test_config_file:
            config_dict = json.load(test_config_file)

        # Override the feature store configuration to use static test config with env vars
        test_config_dir = os.path.join(os.path.dirname(__file__), "../resources")
        config_dict["featureStoreConfig"] = {
            "enabled": True,
            "configDirectory": test_config_dir,  # Use static config directory
            "entityName": config_fixture.feature_store_config.entity_name,
            "ttlDays": config_fixture.feature_store_config.ttl_days,
            "onlineEnabled": True,
            "serviceName": config_fixture.feature_store_config.service_name,
            "serviceVersion": config_fixture.feature_store_config.service_version,
            "offlineRepoStrategy": "local",
            "localRepoConfig": {"repoPath": str(temp_feast_repo)},
            "s3RepoConfig": {
                "bucketName": "drl-trading-features-test",
                "prefix": "features",
                "endpointUrl": None,
                "region": "us-east-1",
                "accessKeyId": None,
                "secretAccessKey": None,
            },
        }

        # Override the features configuration to include test features
        config_dict["featuresConfig"] = {
            "datasetDefinitions": {"EURUSD": ["1h"]},
            "featureDefinitions": [
                {
                    "name": "rsi",
                    "enabled": True,
                    "derivatives": [],
                    "parameterSets": [{"enabled": True, "period": 14}],
                },
                {
                    "name": "close_price",
                    "enabled": True,
                    "derivatives": [],
                    "parameterSets": [],
                },
                {
                    "name": "reward",
                    "enabled": True,
                    "derivatives": [],
                    "parameterSets": [],
                },
            ],
        }

        json.dump(config_dict, config_file, indent=2)

    # Create injector with the real Feast configuration and CORE test modules only
    app_module = CoreModule(config_path=config_path)
    injector = Injector([app_module])

    # Override the feature factory with our test implementation
    # This maintains the dependency boundary - core tests don't depend on strategy
    injector.binder.bind(IFeatureFactory, to=test_feature_factory)

    # Initialize the feature manager with test features
    from drl_trading_core.preprocess.feature.feature_manager import FeatureManager

    feature_manager = injector.get(FeatureManager)
    feature_manager.initialize_features()

    # Store the config path on the injector so we can clean it up later
    # We need to use a workaround for type checking since Injector doesn't have this attribute
    injector._test_config_path = config_path

    return injector


@pytest.fixture(scope="function")
def integration_container(
    real_feast_container: Injector, clean_integration_environment: None
) -> Generator[Injector, None, None]:
    """Provide container with clean integration environment for each test.

    This ensures each test gets a fresh, clean state while using real
    services created through dependency injection.

    Args:
        real_feast_container: DI container with real services
        clean_integration_environment: Clean environment setup

    Returns:
        Injector: Ready-to-use DI container for integration testing
    """
    yield real_feast_container

    # Cleanup the temporary config file if it exists
    config_path = getattr(real_feast_container, "_test_config_path", None)
    if config_path:
        try:
            os.unlink(config_path)
        except OSError:
            pass  # File might already be deleted


@pytest.fixture(scope="function")
def sample_trading_features_df() -> DataFrame:
    """Create comprehensive sample trading features DataFrame for integration testing.

    This fixture provides a realistic set of trading features with proper column names
    and data types that match what the feature store repositories expect.
    """
    # Create realistic time series data with UTC timezone to avoid Feast timezone issues
    timestamps = pd.date_range(
        start="2024-01-01 09:00:00",
        periods=50,
        freq="h",
        tz="UTC",  # Add UTC timezone to match Feast expectations
    )

    # Generate realistic trading feature data
    return DataFrame(
        {
            "event_timestamp": timestamps,
            "symbol": ["EURUSD"] * len(timestamps),
            # Technical indicators - observation space features (match sub-feature names)
            "rsi_14_A1b2c3": [30.0 + (i % 40) + (i * 0.5) for i in range(len(timestamps))],
            # OHLCV data - observation space features (match sub-feature names)
            "close_price": [
                1.0850 + (i % 20) * 0.0001 for i in range(len(timestamps))
            ],  # Fixed: close -> close_1
            # Reward engineering features
            "reward_reward": [0.01 * (i % 20 - 10) for i in range(len(timestamps))],
            "reward_cumulative_return": [0.001 * i for i in range(len(timestamps))],
        }
    )


# S3/MinIO TestContainer Fixtures for Integration Testing


@pytest.fixture(scope="session")
def minio_container() -> Generator[MinioContainer, None, None]:
    """
    Start a MinIO container for S3-compatible testing.

    MinIO is lighter and faster than LocalStack for pure S3 testing.
    """
    with MinioContainer() as minio:
        yield minio


@pytest.fixture
def s3_client_minio(minio_container: MinioContainer) -> boto3.client:
    """Create a boto3 S3 client connected to MinIO container."""
    return boto3.client(
        "s3",
        endpoint_url=f"http://{minio_container.get_container_host_ip()}:{minio_container.get_exposed_port(9000)}",
        aws_access_key_id=minio_container.access_key,
        aws_secret_access_key=minio_container.secret_key,
        region_name="us-east-1",
    )


@pytest.fixture
def s3_test_bucket(s3_client_minio: boto3.client) -> str:
    """Create a test bucket for feature storage testing."""
    import uuid

    # Use a unique bucket name per test session to ensure isolation
    bucket_name = f"test-feature-store-{uuid.uuid4().hex[:8]}"

    # Try to create bucket, ignore if it already exists
    try:
        s3_client_minio.create_bucket(Bucket=bucket_name)
    except s3_client_minio.exceptions.BucketAlreadyOwnedByYou:
        # Bucket already exists, which is fine for our tests
        pass
    except s3_client_minio.exceptions.BucketAlreadyExists:
        # Bucket exists but owned by someone else (shouldn't happen in MinIO)
        pass

    return bucket_name


@pytest.fixture
def s3_feature_store_config(
    s3_client_minio: boto3.client,
    s3_test_bucket: str,
    minio_container: MinioContainer,
    temp_feast_repo: str,
) -> FeatureStoreConfig:
    """Create FeatureStoreConfig for S3 testing."""
    s3_config = S3RepoConfig(
        bucket_name=s3_test_bucket,
        prefix="features",
        endpoint_url=f"http://{minio_container.get_container_host_ip()}:{minio_container.get_exposed_port(9000)}",
        region="us-east-1",
        access_key_id=minio_container.access_key,
        secret_access_key=minio_container.secret_key,
    )

    return FeatureStoreConfig(
        cache_enabled=True,
        entity_name="test_entity",
        ttl_days=30,
        online_enabled=False,
        service_name="test_service",
        service_version="1.0.0",
        offline_repo_strategy=OfflineRepoStrategyEnum.S3,
        s3_repo_config=s3_config,
        config_directory=temp_feast_repo,
    )


@pytest.fixture
def local_feature_store_config(temp_feast_repo: str) -> FeatureStoreConfig:
    """Create FeatureStoreConfig for local filesystem testing."""
    import os
    local_config = LocalRepoConfig(repo_path=os.path.join(temp_feast_repo, "data"))

    return FeatureStoreConfig(
        cache_enabled=True,
        entity_name="test_entity",
        ttl_days=30,
        online_enabled=False,
        service_name="test_service",
        service_version="1.0.0",
        offline_repo_strategy=OfflineRepoStrategyEnum.LOCAL,
        local_repo_config=local_config,
        config_directory=temp_feast_repo,
    )


@pytest.fixture
def s3_integration_container(
    s3_feature_store_config: FeatureStoreConfig,
    test_feature_factory: TestFeatureFactory,
    clean_integration_environment: None,
    minio_container: MinioContainer,
    s3_client_minio: boto3.client,
    s3_test_bucket: str,
) -> Generator[Injector, None, None]:
    """Create a dependency injection container specifically for S3 integration testing.

    This follows the updated Feast initialization pattern with proper S3 configuration.
    Creates a temporary feature_store.yaml with S3 offline store configuration.

    Args:
        s3_feature_store_config: S3-specific feature store configuration
        test_feature_factory: Test feature factory for integration testing
        clean_integration_environment: Clean environment setup
        minio_container: MinIO container for S3-compatible testing
        s3_client_minio: Boto3 S3 client connected to MinIO
        s3_test_bucket: Test bucket for feature storage

    Returns:
        Injector: DI container configured for S3 integration testing
    """
    from drl_trading_core.common.di.core_module import CoreModule

    # Create a temporary directory for S3 Feast configuration
    temp_feast_dir = tempfile.mkdtemp(prefix="feast_s3_integration_")

    # Create the 'test' subdirectory to match STAGE-based convention
    stage_dir = os.path.join(temp_feast_dir, "test")
    os.makedirs(stage_dir, exist_ok=True)

    # Create S3-specific feature_store.yaml in the stage directory
    feast_config_content = f"""project: ai_trading_s3_test
registry:
  path: ${{FEAST_REGISTRY_PATH}}
provider: local
online_store:
  type: sqlite
  path: ${{FEAST_ONLINE_STORE_PATH}}
offline_store:
  type: s3
  path: s3://{s3_test_bucket}/features
  s3_endpoint_url: {s3_feature_store_config.s3_repo_config.endpoint_url}
entity_key_serialization_version: 3
"""

    feast_config_path = os.path.join(stage_dir, "feature_store.yaml")
    with open(feast_config_path, "w") as f:
        f.write(feast_config_content)

    # Set environment variables for S3 credentials (required by pyarrow/Feast)
    original_env = {}
    s3_env_vars = {
        "AWS_ACCESS_KEY_ID": s3_feature_store_config.s3_repo_config.access_key_id,
        "AWS_SECRET_ACCESS_KEY": s3_feature_store_config.s3_repo_config.secret_access_key,
        "AWS_DEFAULT_REGION": s3_feature_store_config.s3_repo_config.region,
        "AWS_ENDPOINT_URL_S3": s3_feature_store_config.s3_repo_config.endpoint_url,
    }

    for key, value in s3_env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    # Create temp config with S3-specific overrides
    temp_fd, config_path = tempfile.mkstemp(suffix=".json", text=True)

    with os.fdopen(temp_fd, "w") as config_file:
        # Load base test config
        base_config_path = os.path.join(
            os.path.dirname(__file__), "../resources/applicationConfig-test.json"
        )
        with open(base_config_path, "r") as base_config_file:
            config_dict = json.load(base_config_file)

        # Override with S3-specific configuration
        config_dict["featureStoreConfig"] = {
            "enabled": True,
            "configDirectory": temp_feast_dir,  # Use temp directory with S3 config
            "entityName": s3_feature_store_config.entity_name,
            "ttlDays": s3_feature_store_config.ttl_days,
            "onlineEnabled": s3_feature_store_config.online_enabled,
            "serviceName": s3_feature_store_config.service_name,
            "serviceVersion": s3_feature_store_config.service_version,
            "offlineRepoStrategy": "s3",
            "s3RepoConfig": {
                "bucketName": s3_feature_store_config.s3_repo_config.bucket_name,
                "prefix": s3_feature_store_config.s3_repo_config.prefix,
                "endpointUrl": s3_feature_store_config.s3_repo_config.endpoint_url,
                "region": s3_feature_store_config.s3_repo_config.region,
                "accessKeyId": s3_feature_store_config.s3_repo_config.access_key_id,
                "secretAccessKey": s3_feature_store_config.s3_repo_config.secret_access_key,
            },
        }

        # Test features configuration
        config_dict["featuresConfig"] = {
            "datasetDefinitions": {"EURUSD": ["1h"]},
            "featureDefinitions": [
                {
                    "name": "rsi",
                    "enabled": True,
                    "derivatives": [],
                    "parameterSets": [{"enabled": True, "period": 14}],
                },
                {
                    "name": "close_price",
                    "enabled": True,
                    "derivatives": [],
                    "parameterSets": [],
                },
                {
                    "name": "reward",
                    "enabled": True,
                    "derivatives": [],
                    "parameterSets": [],
                },
            ],
        }

        json.dump(config_dict, config_file, indent=2)

    try:
        # Create injector with S3 configuration
        app_module = CoreModule(config_path=config_path)
        injector = Injector([app_module])

        # Override feature factory
        injector.binder.bind(IFeatureFactory, to=test_feature_factory)

        # Initialize Feast repository for S3 configurations
        _initialize_feast_repository(temp_feast_dir)

        # Initialize feature manager
        from drl_trading_core.preprocess.feature.feature_manager import FeatureManager

        feature_manager = injector.get(FeatureManager)
        feature_manager.initialize_features()

        # Store paths for cleanup
        injector._test_config_path = config_path
        injector._temp_feast_dir = temp_feast_dir

        yield injector

    finally:
        # Cleanup environment variables
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value

        # Cleanup temporary files
        try:
            os.unlink(config_path)
        except OSError:
            pass
        try:
            shutil.rmtree(temp_feast_dir, ignore_errors=True)
        except OSError:
            pass


@pytest.fixture(params=["local", "s3"])
def parametrized_feature_store_config(
    request,
    local_feature_store_config: FeatureStoreConfig,
    s3_feature_store_config: FeatureStoreConfig,
) -> FeatureStoreConfig:
    """Parametrized fixture that provides both local and S3 feature store configurations."""
    if request.param == "local":
        return local_feature_store_config
    elif request.param == "s3":
        return s3_feature_store_config
    else:
        raise ValueError(f"Unknown parameter: {request.param}")


@pytest.fixture
def parametrized_integration_container(
    parametrized_feature_store_config: FeatureStoreConfig,
    config_fixture: ApplicationConfig,
    test_feature_factory: TestFeatureFactory,
    clean_integration_environment: None,
) -> Generator[Injector, None, None]:
    """Create integration container with parametrized offline repository strategy."""
    from drl_trading_core.common.di.core_module import CoreModule

    # Ensure STAGE is set for consistent testing
    os.environ.setdefault("STAGE", "test")

    # Override the feature store config in the application config
    config_fixture.feature_store_config = parametrized_feature_store_config

    # Initialize Feast repository for local configurations
    if (
        parametrized_feature_store_config.offline_repo_strategy
        == OfflineRepoStrategyEnum.LOCAL
        and parametrized_feature_store_config.local_repo_config is not None
    ):
        repo_path = parametrized_feature_store_config.local_repo_config.repo_path
        _initialize_feast_repository(repo_path)

    # Convert to dict for JSON serialization with JSON-safe mode
    config_dict = config_fixture.model_dump(mode="json")

    # Create temporary config file
    config_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    try:
        json.dump(config_dict, config_file, indent=2)
        config_file.flush()
        config_path = config_file.name
    finally:
        config_file.close()

    try:
        # Create the injector container with the temporary config
        module = CoreModule(config_path=config_path)
        container = Injector([module])

        # Override the feature factory with our test implementation
        # This is essential to prevent abstract interface instantiation errors
        container.binder.bind(IFeatureFactory, to=test_feature_factory)

        # Initialize the feature manager with test features
        from drl_trading_core.preprocess.feature.feature_manager import FeatureManager

        feature_manager = container.get(FeatureManager)
        feature_manager.initialize_features()

        yield container
    finally:
        # Clean up the temporary config file
        try:
            os.unlink(config_path)
        except OSError:
            pass  # File might already be deleted
