import pytest
from datetime import datetime
from unittest.mock import Mock
from drl_trading_adapter.infrastructure.di.adapter_module import AdapterModule
from injector import Injector, Binder, Module, provider, singleton

from drl_trading_common.config.feature_config import (
    FeatureStoreConfig,
)
from drl_trading_common.config.infrastructure_config import DatabaseConfig
from drl_trading_core.core.service.feature.feature_factory_interface import IFeatureFactory
from drl_trading_core.core.port.technical_indicator_service_port import (
    ITechnicalIndicatorServicePort,
)
from drl_trading_core.infrastructure.di.core_module import CoreModule
from drl_trading_preprocess.core.port.message_publisher_port import StoreResampledDataMessagePublisherPort
from drl_trading_preprocess.core.port.preprocessing_message_publisher_port import (
    PreprocessingMessagePublisherPort,
)
from drl_trading_preprocess.application.config.preprocess_config import (
    PreprocessConfig,
    ResampleConfig,
)
from drl_trading_preprocess.application.di.preprocess_module import PreprocessModule


class TestBindingsModule(Module):
    """Test-specific DI bindings for integration tests."""

    def __init__(
        self,
        preprocessing_message_publisher: PreprocessingMessagePublisherPort,
        resampling_message_publisher: StoreResampledDataMessagePublisherPort,
        database_config: DatabaseConfig,
        feature_factory: IFeatureFactory,
        indicator_facade: ITechnicalIndicatorServicePort,
    ):
        self._preprocessing_message_publisher = preprocessing_message_publisher
        self._resampling_message_publisher = resampling_message_publisher
        self._database_config = database_config
        self._feature_factory = feature_factory
        self._indicator_facade = indicator_facade

    def configure(self, binder: Binder) -> None:  # type: ignore[override]
        """Configure test-specific bindings."""
        pass  # Use provider methods instead

    @provider  # type: ignore[misc]
    @singleton
    def provide_preprocessing_message_publisher(self) -> PreprocessingMessagePublisherPort:
        """Provide the test preprocessing message publisher instance."""
        return self._preprocessing_message_publisher

    @provider  # type: ignore[misc]
    @singleton
    def provide_resampling_message_publisher(self) -> StoreResampledDataMessagePublisherPort:
        """Provide the test resampling message publisher instance."""
        return self._resampling_message_publisher

    @provider  # type: ignore[misc]
    @singleton
    def provide_database_config(self) -> DatabaseConfig:
        """Provide the test database configuration."""
        return self._database_config

    @provider  # type: ignore[misc]
    @singleton
    def provide_feature_factory(self) -> IFeatureFactory:
        """Provide the test feature factory."""
        return self._feature_factory

    @provider  # type: ignore[misc]
    @singleton
    def provide_indicator_facade(self) -> ITechnicalIndicatorServicePort:
        """Provide the mock indicator facade."""
        return self._indicator_facade


@pytest.fixture(scope="function")
def spy_resampling_message_publisher() -> StoreResampledDataMessagePublisherPort:
    """Create a mock resampling message publisher for verification.

    Returns:
        Mock: Mock publisher for resampling notifications (spec=MessagePublisherPort)
    """
    return Mock(spec=StoreResampledDataMessagePublisherPort)


@pytest.fixture(scope="function")
def test_preprocess_config(feature_store_config: FeatureStoreConfig) -> PreprocessConfig:
    """Create test preprocess configuration with state persistence disabled.

    Args:
        feature_store_config: Feature store configuration

    Returns:
        PreprocessConfig: Test configuration with optimized settings
    """
    from drl_trading_preprocess.application.config.preprocess_config import DaskConfigs, FeatureComputationConfig
    from drl_trading_common.config.dask_config import DaskConfig
    from drl_trading_common.config.kafka_config import KafkaConsumerConfig

    # Create ResampleConfig with state persistence disabled for testing
    resample_config = ResampleConfig(
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

    # Create DaskConfigs for tests (synchronous scheduler for deterministic behavior)
    dask_configs = DaskConfigs(
        coverage_analysis=DaskConfig(
            scheduler="synchronous",
            num_workers=1,
            threads_per_worker=1,
            memory_limit_per_worker_mb=512,
        ),
        feature_computation=DaskConfig(
            scheduler="synchronous",
            num_workers=1,
            threads_per_worker=1,
            memory_limit_per_worker_mb=512,
        ),
    )

    # Create KafkaConsumerConfig for tests (no actual consumption in integration tests)
    kafka_consumers = KafkaConsumerConfig(
        consumer_group_id="test-preprocess-consumer-group",
        topic_subscriptions=[],
    )

    return PreprocessConfig(
        feature_store_config=feature_store_config,
        feature_computation_config=FeatureComputationConfig(warmup_candles=10),
        resample_config=resample_config,
        dask_configs=dask_configs,
        kafka_consumers=kafka_consumers,
    )


@pytest.fixture(scope="function")
def real_feast_container(
    temp_feast_repo: str,
    spy_message_publisher: PreprocessingMessagePublisherPort,
    spy_resampling_message_publisher: StoreResampledDataMessagePublisherPort,
    database_config: DatabaseConfig,
    test_feature_factory: IFeatureFactory,
    mock_indicator_facade: ITechnicalIndicatorServicePort,
    test_preprocess_config: PreprocessConfig,
) -> Injector:
    """Create a dependency injection container with REAL Feast integration.

    This fixture provides a configured injector instance with real services
    for true integration testing. Includes test-specific bindings for:
    - Spy preprocessing message publisher for verification
    - Spy resampling message publisher for verification
    - Test database configuration from testcontainers
    - Test feature factory for creating test features
    - Mock indicator facade for test features

    Args:
        temp_feast_repo: Path to the temporary Feast repository
        spy_message_publisher: Spy publisher for preprocessing message verification
        spy_resampling_message_publisher: Spy publisher for resampling message verification
        database_config: Database config from testcontainer
        test_feature_factory: Test feature factory
        mock_indicator_facade: Mock indicator facade
        test_preprocess_config: Test preprocess configuration

    Returns:
        Injector: Configured DI container with real services and test bindings
    """
    # Create test bindings module
    test_bindings = TestBindingsModule(
        preprocessing_message_publisher=spy_message_publisher,
        resampling_message_publisher=spy_resampling_message_publisher,
        database_config=database_config,
        feature_factory=test_feature_factory,
        indicator_facade=mock_indicator_facade,
    )

    # Create injector with the real Feast configuration and test bindings
    app_module = AdapterModule()
    core_module = CoreModule()
    preprocess_module = PreprocessModule(test_preprocess_config)
    injector = Injector([app_module, preprocess_module, core_module, test_bindings])

    return injector
