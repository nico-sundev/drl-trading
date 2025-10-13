import pytest
from drl_trading_adapter.infrastructure.di.adapter_module import AdapterModule
from injector import Injector, Binder, Module, provider, singleton

from drl_trading_common.config.feature_config import (
    FeatureStoreConfig,
)
from drl_trading_common.config.infrastructure_config import DatabaseConfig
from drl_trading_common.interface.feature.feature_factory_interface import IFeatureFactory
from drl_trading_common.interface.indicator.technical_indicator_facade_interface import (
    ITechnicalIndicatorFacade,
)
from drl_trading_core.infrastructure.di.core_module import CoreModule
from drl_trading_preprocess.adapter.messaging.publisher.stub_message_publisher import (
    StubMessagePublisher,
)
from drl_trading_preprocess.core.port.message_publisher_port import MessagePublisherPort
from drl_trading_preprocess.core.port.preprocessing_message_publisher_port import (
    PreprocessingMessagePublisherPort,
)
from drl_trading_preprocess.infrastructure.config.preprocess_config import (
    PreprocessConfig,
    ResampleConfig,
)
from drl_trading_preprocess.infrastructure.di.preprocess_module import PreprocessModule


class TestBindingsModule(Module):
    """Test-specific DI bindings for integration tests."""

    def __init__(
        self,
        preprocessing_message_publisher: PreprocessingMessagePublisherPort,
        resampling_message_publisher: MessagePublisherPort,
        database_config: DatabaseConfig,
        feature_factory: IFeatureFactory,
        indicator_facade: ITechnicalIndicatorFacade,
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
    def provide_resampling_message_publisher(self) -> MessagePublisherPort:
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
    def provide_indicator_facade(self) -> ITechnicalIndicatorFacade:
        """Provide the mock indicator facade."""
        return self._indicator_facade


@pytest.fixture(scope="function")
def spy_resampling_message_publisher() -> MessagePublisherPort:
    """Create a spy resampling message publisher for verification.

    Returns:
        StubMessagePublisher: Publisher for resampling notifications
    """
    return StubMessagePublisher(log_level="DEBUG")


@pytest.fixture(scope="function")
def test_preprocess_config(feature_store_config: FeatureStoreConfig) -> PreprocessConfig:
    """Create test preprocess configuration with state persistence disabled.

    Args:
        feature_store_config: Feature store configuration

    Returns:
        PreprocessConfig: Test configuration with optimized settings
    """
    # Create ResampleConfig with state persistence disabled for testing
    resample_config = ResampleConfig(state_persistence_enabled=False)

    return PreprocessConfig(
        feature_store_config=feature_store_config,
        resample_config=resample_config,
    )


@pytest.fixture(scope="function")
def real_feast_container(
    temp_feast_repo: str,
    spy_message_publisher: PreprocessingMessagePublisherPort,
    spy_resampling_message_publisher: MessagePublisherPort,
    database_config: DatabaseConfig,
    test_feature_factory: IFeatureFactory,
    mock_indicator_facade: ITechnicalIndicatorFacade,
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
