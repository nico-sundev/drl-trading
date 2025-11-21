"""Dependency injection module for preprocess service (config injected)."""

import logging

from drl_trading_common.adapter.messaging.kafka_producer_adapter import (
    KafkaProducerAdapter,
)
from drl_trading_common.config.feature_config import FeatureStoreConfig
from drl_trading_common.config.infrastructure_config import DatabaseConfig
from drl_trading_common.messaging.kafka_handler_registry import KafkaHandlerRegistry
from drl_trading_preprocess.adapter.messaging.kafka_handler_constants import (
    HANDLER_ID_PREPROCESSING_REQUEST,
)
from drl_trading_preprocess.adapter.messaging.kafka_message_handler_factory import (
    KafkaMessageHandlerFactory,
)
from drl_trading_preprocess.adapter.messaging.publisher.kafka_message_publisher import (
    KafkaMessagePublisher,
)
from drl_trading_preprocess.adapter.messaging.publisher.kafka_preprocessing_message_publisher import (
    KafkaPreprocessingMessagePublisher,
)
from drl_trading_preprocess.adapter.feature_store.feature_store_save_repository import (
    FeatureStoreSaveRepository,
)
from drl_trading_preprocess.core.orchestrator.preprocessing_orchestrator import (
    PreprocessingOrchestrator,
)
from drl_trading_preprocess.core.port.feature_store_save_port import (
    IFeatureStoreSavePort,
)
from drl_trading_preprocess.core.port.message_publisher_port import StoreResampledDataMessagePublisherPort
from drl_trading_preprocess.core.port.preprocessing_message_publisher_port import (
    PreprocessingMessagePublisherPort,
)
from drl_trading_preprocess.core.port.state_persistence_port import (
    IStatePersistencePort,
)
from drl_trading_preprocess.adapter.resampling.noop_state_persistence_service import (
    NoOpStatePersistenceService,
)
from drl_trading_preprocess.adapter.resampling.state_persistence_service import (
    StatePersistenceService,
)
from drl_trading_preprocess.infrastructure.config.preprocess_config import (
    PreprocessConfig,
    DaskConfigs,
    ResampleConfig,
    FeatureComputationConfig as PreprocessFeatureComputationConfig,
)
from drl_trading_core.core.config.feature_computation_config import (
    FeatureComputationConfig,
)
from drl_trading_preprocess.infrastructure.config.resilience_constants import (
    RETRY_CONFIG_KAFKA_DLQ,
    RETRY_CONFIG_KAFKA_PREPROCESSING_COMPLETED,
    RETRY_CONFIG_KAFKA_RESAMPLED_DATA,
)
from injector import Binder, Module, provider, singleton


logger = logging.getLogger(__name__)


class PreprocessModule(Module):
    """Dependency injection module for preprocess service.

    Expects the already-loaded config instance to be passed from bootstrap.
    """

    def __init__(self, config: PreprocessConfig) -> None:
        self._config = config

    def configure(self, binder: Binder) -> None:  # type: ignore[override]
        binder.bind(
            IFeatureStoreSavePort,
            to=FeatureStoreSaveRepository,
            scope=singleton,
        )
        # Prevent auto-wiring of StatePersistenceService - use provider instead
        # This ensures Optional[StatePersistenceService] doesn't try to auto-instantiate

    @provider
    @singleton
    def provide_preprocess_config(self) -> PreprocessConfig:
        """Provide preprocess configuration (no reload)."""
        return self._config

    @provider
    @singleton
    def provide_feature_store_config(self) -> FeatureStoreConfig:
        """Provide feature store configuration (no reload)."""
        return self._config.feature_store_config

    @provider
    @singleton
    def provide_dask_configs(self) -> DaskConfigs:
        """Provide Dask configurations collection for parallel processing."""
        return self._config.dask_configs

    @provider
    @singleton
    def provide_resample_config(self) -> ResampleConfig:
        """Provide resample configuration."""
        return self._config.resample_config

    @provider
    @singleton
    def provide_preprocess_feature_computation_config(self) -> PreprocessFeatureComputationConfig:
        """Provide preprocess feature computation configuration."""
        return self._config.feature_computation_config

    @provider
    @singleton
    def provide_database_config(self) -> DatabaseConfig:
        """Provide database configuration."""
        return self._config.infrastructure.database

    @provider
    @singleton
    def provide_feature_computation_config(self, dask_configs: DaskConfigs) -> FeatureComputationConfig:
        """
        Provide feature computation configuration for FeatureManager.

        Extracts the feature_computation Dask config from the service's DaskConfigs
        collection and wraps it in a FeatureComputationConfig for injection into
        the core FeatureManager.

        Args:
            dask_configs: The service's collection of Dask configurations

        Returns:
            FeatureComputationConfig with the feature_computation Dask settings
        """
        return FeatureComputationConfig(dask=dask_configs.feature_computation)

    @provider  # type: ignore[misc]
    @singleton
    def provide_state_persistence_service(self) -> IStatePersistencePort:
        """
        Provide state persistence service based on configuration.

        Returns:
            StatePersistenceService if enabled, NoOpStatePersistenceService if disabled

        Note: Always returns a valid implementation (Null Object Pattern).
        When disabled, returns no-op implementation that safely does nothing.
        """
        if not self._config.resample_config.state_persistence_enabled:
            return NoOpStatePersistenceService()

        return StatePersistenceService(
            state_file_path=self._config.resample_config.state_file_path,
            backup_interval=self._config.resample_config.state_backup_interval
        )

    @provider
    @singleton
    def provide_kafka_handler_registry(self, orchestrator: PreprocessingOrchestrator) -> KafkaHandlerRegistry:
        """Register all Kafka message handlers for this service."""
        factory = KafkaMessageHandlerFactory()
        handlers = {
            HANDLER_ID_PREPROCESSING_REQUEST: factory.create_preprocessing_request_handler(orchestrator),
        }
        return KafkaHandlerRegistry(handlers)

    @provider
    @singleton
    def provide_message_publisher(self) -> StoreResampledDataMessagePublisherPort:
        """
        Provide MessagePublisherPort implementation with Kafka producers.

        Creates two separate KafkaProducerAdapter instances:
        - One for resampled data with aggressive retry config
        - One for DLQ with minimal retry config
        """
        if not self._config.infrastructure.kafka:
            raise ValueError("Kafka configuration is required for message publishing")

        if not self._config.infrastructure.resilience:
            raise ValueError("Resilience configuration is required for message publishing")

        if not self._config.kafka_topics:
            raise ValueError("Kafka topics configuration is required for message publishing")

        # Get Kafka producer configuration
        producer_config = self._config.infrastructure.kafka.get_producer_config()

        # Get retry configurations from resilience config
        resampled_data_retry = self._config.infrastructure.resilience.get_retry_config(
            RETRY_CONFIG_KAFKA_RESAMPLED_DATA
        )
        dlq_retry = self._config.infrastructure.resilience.get_retry_config(
            RETRY_CONFIG_KAFKA_DLQ
        )

        # Get topic names from service config
        resampled_data_topic = self._config.kafka_topics.resampled_data.topic
        error_topic = self._config.kafka_topics.preprocessing_error.topic

        # Create producer adapters with appropriate retry configs
        resampled_data_producer = KafkaProducerAdapter(
            producer_config=producer_config,
            retry_config=resampled_data_retry,
            dlq_topic=error_topic,  # Failed resampled data goes to preprocessing_error DLQ
        )

        error_producer = KafkaProducerAdapter(
            producer_config=producer_config,
            retry_config=dlq_retry,
            dlq_topic=None,  # DLQ producer doesn't have a DLQ (avoid infinite loop)
        )

        return KafkaMessagePublisher(
            resampled_data_producer=resampled_data_producer,
            error_producer=error_producer,
            resampled_data_topic=resampled_data_topic,
            error_topic=error_topic,
        )

    @provider
    @singleton
    def provide_preprocessing_message_publisher(self) -> PreprocessingMessagePublisherPort:
        """
        Provide PreprocessingMessagePublisherPort implementation with Kafka producers.

        Creates two separate KafkaProducerAdapter instances:
        - One for preprocessing completion events with moderate retry config
        - One for DLQ with minimal retry config
        """
        if not self._config.infrastructure.kafka:
            raise ValueError("Kafka configuration is required for preprocessing message publishing")

        if not self._config.infrastructure.resilience:
            raise ValueError("Resilience configuration is required for preprocessing message publishing")

        if not self._config.kafka_topics:
            raise ValueError("Kafka topics configuration is required for preprocessing message publishing")

        # Get Kafka producer configuration
        producer_config = self._config.infrastructure.kafka.get_producer_config()

        # Get retry configurations from resilience config
        completion_retry = self._config.infrastructure.resilience.get_retry_config(
            RETRY_CONFIG_KAFKA_PREPROCESSING_COMPLETED
        )
        dlq_retry = self._config.infrastructure.resilience.get_retry_config(
            RETRY_CONFIG_KAFKA_DLQ
        )

        # Get topic names from service config
        completion_topic = self._config.kafka_topics.preprocessing_completed.topic
        error_topic = self._config.kafka_topics.preprocessing_error.topic

        # Create producer adapters with appropriate retry configs
        completion_producer = KafkaProducerAdapter(
            producer_config=producer_config,
            retry_config=completion_retry,
            dlq_topic=error_topic,  # Failed completions go to preprocessing_error DLQ
        )

        error_producer = KafkaProducerAdapter(
            producer_config=producer_config,
            retry_config=dlq_retry,
            dlq_topic=None,  # DLQ producer doesn't have a DLQ (avoid infinite loop)
        )

        return KafkaPreprocessingMessagePublisher(
            completion_producer=completion_producer,
            error_producer=error_producer,
            completion_topic=completion_topic,
            error_topic=error_topic,
        )
