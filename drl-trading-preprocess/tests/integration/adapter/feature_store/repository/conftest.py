import os
from typing import Generator, Optional

import boto3
import pandas as pd
import pytest
from drl_trading_adapter.infrastructure.di.adapter_module import AdapterModule
from drl_trading_core.core.port.base_feature import BaseFeature
from drl_trading_common.adapter.model.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_common.config.feature_config import (
    FeatureStoreConfig,
    LocalRepoConfig,
    S3RepoConfig,
)
from drl_trading_core.core.service.feature.decorator.feature_role_decorator import feature_role
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.enum.offline_repo_strategy_enum import OfflineRepoStrategyEnum
from drl_trading_core.core.port.technical_indicator_service_port import (
    ITechnicalIndicatorServicePort,
)
from drl_trading_common.core.model.dataset_identifier import DatasetIdentifier
from drl_trading_common.core.model.timeframe import Timeframe
from injector import Injector
from pandas import DataFrame
from testcontainers.minio import MinioContainer


from drl_trading_preprocess.infrastructure.config.preprocess_config import PreprocessConfig
from drl_trading_preprocess.infrastructure.di.preprocess_module import PreprocessModule
from drl_trading_core.core.dto.feature_view_metadata import FeatureViewMetadata
from drl_trading_core.core.model.feature_config_version_info import FeatureConfigVersionInfo
from datetime import datetime


# Test Feature Implementations for Core Integration Testing
class MockTechnicalIndicatorFacade(ITechnicalIndicatorServicePort):
    """Mock technical indicator facade that returns controlled test data for integration testing."""

    def __init__(self) -> None:
        self._indicators: dict[str, DataFrame] = {}

    def register_instance(self, name: str, indicator_type, **params) -> None:
        """Register a mock indicator that returns predictable test data."""
        # Create predictable test data based on indicator type with UTC timezone
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1h", tz="UTC")  # Use lowercase 'h' as 'H' is deprecated

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

    def _call_indicator_backend(self, method_call) -> Optional[DataFrame]:
        """Mock implementation."""
        return None

    def _get_feature_name(self) -> str:
        return self._feature_name

    def _get_sub_features_names(self) -> list[str]:
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

    def _get_config_to_string(self) -> str:
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

    def _call_indicator_backend(self, method_call) -> Optional[DataFrame]:
        """Mock implementation."""
        return None

    def _get_feature_name(self) -> str:
        return self._feature_name

    def _get_sub_features_names(self) -> list[str]:
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

    def _get_config_to_string(self) -> Optional[str]:
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

    def _call_indicator_backend(self, method_call) -> Optional[DataFrame]:
        """Mock implementation."""
        return None

    def _get_feature_name(self) -> str:
        return self._feature_name

    def _get_sub_features_names(self) -> list[str]:
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

    def _get_config_to_string(self) -> Optional[str]:
        return None

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
        freq="h",  # Use lowercase 'h' as 'H' is deprecated in newer pandas versions
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

    # Use the actual test config directory that contains feature_store.yaml
    test_config_dir = os.path.join(os.path.dirname(__file__), "../../../../resources/test")

    return FeatureStoreConfig(
        cache_enabled=True,
        entity_name="test_entity",
        ttl_days=30,
        online_enabled=True,
        service_name="test_service",
        service_version="1.0.0",
        offline_repo_strategy=OfflineRepoStrategyEnum.S3,
        s3_repo_config=s3_config,
        config_directory=test_config_dir,
    )


@pytest.fixture
def local_feature_store_config(temp_feast_repo: str) -> FeatureStoreConfig:
    """Create FeatureStoreConfig for local filesystem testing."""
    import os
    local_config = LocalRepoConfig(repo_path=os.path.join(temp_feast_repo))

    # Use the actual test config directory that contains feature_store.yaml
    test_config_dir = os.path.join(os.path.dirname(__file__), "../../../../resources/test")

    return FeatureStoreConfig(
        cache_enabled=True,
        entity_name="test_entity",
        ttl_days=30,
        online_enabled=True,
        service_name="test_service",
        service_version="1.0.0",
        offline_repo_strategy=OfflineRepoStrategyEnum.LOCAL,
        local_repo_config=local_config,
        config_directory=test_config_dir,
    )


# @pytest.fixture(params=["local", "s3"])
@pytest.fixture(params=["local"])
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
    clean_integration_environment: None,
) -> Generator[Injector, None, None]:
    """Create integration container with parametrized offline repository strategy."""

    # Ensure STAGE is set for consistent testing
    os.environ.setdefault("STAGE", "test")

    # Initialize Feast repository for local configurations
    if (
        parametrized_feature_store_config.offline_repo_strategy
        == OfflineRepoStrategyEnum.LOCAL
        and parametrized_feature_store_config.local_repo_config is not None
    ):
        repo_path = parametrized_feature_store_config.local_repo_config.repo_path
        _initialize_feast_repository(repo_path)

    # Create the injector container with the temporary config
    app_module = AdapterModule()
    from drl_trading_preprocess.infrastructure.config.preprocess_config import FeatureComputationConfig, ResampleConfig
    preprocess_module = PreprocessModule(PreprocessConfig(
        feature_store_config=parametrized_feature_store_config,
        feature_computation_config=FeatureComputationConfig(warmup_candles=10),
        resample_config=ResampleConfig(
            state_persistence_enabled=False,
            historical_start_date=datetime(2020, 1, 1),
            max_batch_size=1000,
            progress_log_interval=10,
            enable_incomplete_candle_publishing=False,
            chunk_size=100,
            memory_warning_threshold_mb=100,
            pagination_limit=1000,
            max_memory_usage_mb=500,
            state_file_path="/tmp/test_state.json",
            state_backup_interval=60,
            auto_cleanup_inactive_symbols=False,
            inactive_symbol_threshold_hours=24,
        )
    ))
    injector = Injector([app_module, preprocess_module])

    yield injector


@pytest.fixture
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


def _create_feature_view_requests(
    feature_version_info: FeatureConfigVersionInfo,
    symbol: str
) -> list[FeatureViewMetadata]:
    """Internal helper to create feature view requests for a specific symbol.

    This is an internal helper function used only by fixtures in this conftest file.
    Do not import or use directly from tests.
    """
    # Create dataset identifier and mock indicator service
    dataset_identifier = DatasetIdentifier(symbol=symbol, timeframe=Timeframe.HOUR_1)
    indicator_service = MockTechnicalIndicatorFacade()

    # Create list of Feature ViewRequestContainers, one per feature
    result = []
    created_features = {}  # Track features we've already created to avoid duplicates

    for feature_def in feature_version_info.feature_definitions:
        if not feature_def.get("enabled", True):
            continue  # Skip disabled features

        role_str = feature_def.get("role", "observation_space")
        role = FeatureRoleEnum[role_str.upper()] if isinstance(role_str, str) else role_str

        # Create concrete feature instances based on feature name, avoiding duplicates
        feature_name = feature_def["name"]
        feature_key = f"{role}_{feature_name}"

        # For reward features, both reward and cumulative_return map to the same TestRewardFeature
        if feature_name in ["reward", "cumulative_return"]:
            reward_key = f"{role}_reward_feature"
            if reward_key not in created_features:
                feature = TestRewardFeature(dataset_identifier, indicator_service)
                created_features[reward_key] = feature
                result.append(FeatureViewMetadata(
                    dataset_identifier=dataset_identifier,
                    feature_metadata=feature.get_metadata(),
                ))
        elif feature_name == "rsi_14":
            if feature_key not in created_features:
                config = TestRsiConfig()
                feature = TestRsiFeature(dataset_identifier, indicator_service, config)
                created_features[feature_key] = feature
                result.append(FeatureViewMetadata(
                    dataset_identifier=dataset_identifier,
                    feature_metadata=feature.get_metadata(),
                ))
        elif feature_name == "close_1":
            if feature_key not in created_features:
                feature = TestClosePriceFeature(dataset_identifier, indicator_service)
                created_features[feature_key] = feature
                result.append(FeatureViewMetadata(
                    dataset_identifier=dataset_identifier,
                    feature_metadata=feature.get_metadata(),
                ))
        else:
            # Default to close price feature for unknown features
            if feature_key not in created_features:
                feature = TestClosePriceFeature(dataset_identifier, indicator_service)
                created_features[feature_key] = feature
                result.append(FeatureViewMetadata(
                    dataset_identifier=dataset_identifier,
                    feature_metadata=feature.get_metadata(),
                ))

    return result


@pytest.fixture
def feature_view_requests_fixture(
    feature_version_info_fixture: FeatureConfigVersionInfo,
) -> list[FeatureViewMetadata]:
    """Create feature view requests from feature version info for integration testing.

    This fixture provides a default EURUSD symbol for tests that don't need multi-symbol testing.
    For tests that need different symbols, use the symbol_feature_view_requests_fixture with parametrization.
    """
    return _create_feature_view_requests(feature_version_info_fixture, symbol="EURUSD")


@pytest.fixture
def symbol_feature_view_requests_fixture(
    request,
    feature_version_info_fixture: FeatureConfigVersionInfo,
) -> list[FeatureViewMetadata]:
    """Parametrized fixture for creating feature view requests with different symbols.

    Use with pytest.mark.parametrize and indirect=True:

    @pytest.mark.parametrize("symbol_feature_view_requests_fixture", ["EURUSD", "GBPUSD"], indirect=True)
    def test_example(symbol_feature_view_requests_fixture):
        # symbol_feature_view_requests_fixture contains requests for the specified symbol
    """
    symbol = getattr(request, 'param', 'EURUSD')  # Default to EURUSD if no param provided
    return _create_feature_view_requests(feature_version_info_fixture, symbol=symbol)
