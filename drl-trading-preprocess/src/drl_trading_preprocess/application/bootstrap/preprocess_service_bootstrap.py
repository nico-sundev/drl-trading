
"""
Preprocess service bootstrap using ServiceBootstrap framework.

Implements the standard service bootstrap pattern with Flask web interface
for health checks while maintaining the preprocessing service's core functionality.

Integrates the standardized T005 logging framework (ServiceLogger) so logs
include class/module names and service context consistently across services.
"""

import logging
import threading
from typing import Dict, List, Optional

from injector import Module

from drl_trading_adapter.application.di.adapter_module import AdapterModule
from drl_trading_common.adapter.messaging.kafka_consumer_topic_adapter import (
    KafkaConsumerTopicAdapter,
)
from drl_trading_common.config.kafka_config import ConsumerFailurePolicy
from drl_trading_common.application.bootstrap.flask_service_bootstrap import FlaskServiceBootstrap
from drl_trading_common.application.health.basic_health_checks import (
    SystemResourcesHealthCheck,
    ServiceStartupHealthCheck,
    ConfigurationHealthCheck,
)
from drl_trading_common.application.health.health_check import HealthCheck
from drl_trading_common.messaging.kafka_handler_registry import KafkaHandlerRegistry
from drl_trading_core.application.di.core_module import CoreModule
from drl_trading_preprocess.application.config.preprocess_config import PreprocessConfig


logger = logging.getLogger(__name__)


class PreprocessServiceBootstrap(FlaskServiceBootstrap):
    """Bootstrap implementation for the preprocess service.

    Responsibilities (infrastructure only):
    - Provide DI modules (fail fast on import issues)
    - Register health checks (system, startup, configuration)
    - Supply route registrar (default health-only endpoints for now)
    - Start/stop placeholder business logic hooks

    Domain logic (feature computation, data validation, etc.) will be delivered
    via DI-managed core services, keeping this class thin per hexagonal design.
    """

    def __init__(self) -> None:
        """Initialize the preprocess service bootstrap."""
        super().__init__(service_name="drl-trading-preprocess", config_class=PreprocessConfig)
        self._startup_health_check = ServiceStartupHealthCheck("preprocess_startup")
        self._kafka_consumer: Optional[KafkaConsumerTopicAdapter] = None
        self._kafka_consumer_thread: Optional[threading.Thread] = None

    def get_dependency_modules(self, app_config: PreprocessConfig) -> List[Module]:
        """Return DI modules using existing loaded config instance.

        The provided app_config avoids redundant config reloads inside DI.
        """
        from drl_trading_preprocess.application.di.preprocess_module import PreprocessModule  # type: ignore
        from drl_trading_strategy_example.infrastructure.di.feature_computation_module import (
            FeatureComputationModule,
        )

        return [
            PreprocessModule(app_config),
            CoreModule(),
            AdapterModule(),
            FeatureComputationModule(),  # Provides features + indicators only (no RL environment)
        ]

    def get_health_checks(self) -> List[HealthCheck]:
        """Return health checks (always includes configuration check)."""
        return [
            SystemResourcesHealthCheck(
                name="preprocess_system_resources",
                cpu_threshold=90.0,
                memory_threshold=90.0,
            ),
            self._startup_health_check,
            ConfigurationHealthCheck(self.config, "preprocess_configuration"),  # type: ignore[arg-type]
        ]

    # Logging is configured by the base ServiceBootstrap._setup_logging

    def _start_service(self) -> None:
        """Start preprocess business logic placeholder.

        Real preprocessing orchestration (feature pipelines, schedulers, etc.)
        will be hooked here. For now, we simply call component initialization
        placeholders and mark startup healthy for health reporting.
        """
        service_logger = logging.getLogger(__name__)
        try:
            service_logger.info("=== STARTING PREPROCESS SERVICE BUSINESS LOGIC ===")
            self._initialize_preprocessing_components()
            self._startup_health_check.mark_startup_completed(success=True)
            service_logger.info(
                "=== PREPROCESS SERVICE BUSINESS LOGIC INITIALIZED SUCCESSFULLY ==="
            )
        except Exception as e:  # pragma: no cover - defensive path
            self._startup_health_check.mark_startup_completed(success=False, error_message=str(e))
            service_logger.error("Failed to start preprocess service: %s", e, exc_info=True)
            raise

    def _stop_service(self) -> None:
        """Stop preprocess service-specific logic."""
        service_logger = logging.getLogger(__name__)
        service_logger.info("=== STOPPING PREPROCESS SERVICE BUSINESS LOGIC ===")
        try:
            # Stop Kafka consumer gracefully
            if self._kafka_consumer:
                service_logger.info("Stopping Kafka consumer...")
                self._kafka_consumer.stop()

                # Wait for consumer thread to finish
                if self._kafka_consumer_thread and self._kafka_consumer_thread.is_alive():
                    service_logger.info("Waiting for Kafka consumer thread to finish...")
                    self._kafka_consumer_thread.join(timeout=10)

                    if self._kafka_consumer_thread.is_alive():
                        service_logger.warning("Kafka consumer thread did not finish within timeout")
                    else:
                        service_logger.info("Kafka consumer thread finished successfully")

                self._kafka_consumer = None
                self._kafka_consumer_thread = None

            service_logger.info("Preprocess service business logic stopped successfully")
        except Exception as e:  # pragma: no cover - defensive
            service_logger.error("Error stopping preprocess service: %s", e, exc_info=True)

    def _initialize_preprocessing_components(self) -> None:
        """Initialize preprocessing-specific components."""
        logger.info("Setting up preprocessing components...")
        self._setup_feature_computation()
        self._setup_data_validation()
        self._setup_feature_store()
        self._setup_messaging()
        logger.info("Preprocessing components initialized")

    def _setup_feature_computation(self) -> None:
        """Setup feature computation pipelines."""
        logger.info("Setting up feature computation (placeholder)...")
        # TODO: Implement feature computation setup

    def _setup_data_validation(self) -> None:
        """Setup data validation pipelines."""
        logger.info("Setting up data validation (placeholder)...")
        # TODO: Implement data validation setup

    def _setup_feature_store(self) -> None:
        """Setup feature store integration."""
        logger.info("Setting up feature store integration (placeholder)...")
        # TODO: Implement feature store setup

    def _setup_messaging(self) -> None:
        """Setup messaging infrastructure."""
        logger.info("Setting up Kafka consumer...")

        config: PreprocessConfig = self.config  # type: ignore[assignment]

        # Validate prerequisites
        if not self._validate_messaging_prerequisites(config):
            return

        # Get dependencies
        handler_registry = self._get_handler_registry()
        if not handler_registry:
            return

        # Build consumer configuration
        if not config.kafka_consumers:
            logger.error("Kafka consumer configuration missing while setting up messaging; skipping Kafka consumer setup")
            return
        consumer_group_id = config.kafka_consumers.consumer_group_id
        consumer_config = config.infrastructure.kafka.get_consumer_config(group_id=consumer_group_id)
        topics = [sub.topic for sub in config.kafka_consumers.topic_subscriptions]

        # Build topic handlers and validate
        topic_handlers = self._build_topic_handlers(config, handler_registry)

        # Build failure policies
        failure_policies = self._build_failure_policies(config)

        # Create DLQ producer if needed
        dlq_producer = self._create_dlq_producer_if_needed(config, failure_policies)

        # Create retry producer if needed
        retry_producer = self._create_retry_producer_if_needed(config, failure_policies)

        # Create and start consumer
        self._create_and_start_consumer(
            consumer_config=consumer_config,
            topics=topics,
            topic_handlers=topic_handlers,
            failure_policies=failure_policies,
            dlq_producer=dlq_producer,
            retry_producer=retry_producer,
            consumer_group_id=consumer_group_id,
            handler_count=len(handler_registry),
        )

    def _validate_messaging_prerequisites(self, config: PreprocessConfig) -> bool:
        """Validate that all prerequisites for messaging setup are met.

        Args:
            config: Service configuration

        Returns:
            True if all prerequisites are met, False otherwise
        """
        if not config.infrastructure.kafka:
            logger.warning("No Kafka configuration found, skipping Kafka consumer setup")
            return False

        if not config.kafka_consumers:
            logger.warning("No Kafka consumer configuration found, skipping Kafka consumer setup")
            return False

        if not self.injector:
            logger.error("DI injector not initialized, cannot setup Kafka consumer")
            return False

        return True

    def _get_handler_registry(self) -> Optional[KafkaHandlerRegistry]:
        """Get Kafka handler registry from DI container.

        Returns:
            Handler registry if available, None on error
        """
        try:
            return self.injector.get(KafkaHandlerRegistry)  # type: ignore[union-attr]
        except Exception as e:
            logger.error(
                f"Failed to get Kafka handler registry from DI container: {e}",
                exc_info=True
            )
            return None

    def _build_topic_handlers(
        self,
        config: PreprocessConfig,
        handler_registry: KafkaHandlerRegistry
    ) -> dict:
        """Build topic to handler mapping and validate all handlers exist.

        Args:
            config: Service configuration
            handler_registry: Registry of available handlers

        Returns:
            Mapping of topic to handler function
        """

        if not config.kafka_consumers:
            # This should not happen due to earlier validation, but guard for static typing and safety.
            raise ValueError("Kafka consumer configuration is missing")
        topic_handlers = {
            sub.topic: handler
            for sub in config.kafka_consumers.topic_subscriptions
            if (handler := handler_registry.get_handler(sub.handler_id)) is not None
        }

        # Validate all handlers were found
        if len(topic_handlers) != len(config.kafka_consumers.topic_subscriptions):
            missing = [
                sub.handler_id
                for sub in config.kafka_consumers.topic_subscriptions
                if handler_registry.get_handler(sub.handler_id) is None
            ]
            raise ValueError(f"No handlers registered for handler_ids: {missing}")

        return topic_handlers

    def _build_failure_policies(self, config: PreprocessConfig) -> dict:
        """Build failure policies mapping from configuration.

        Args:
            config: Service configuration

        Returns:
            Mapping of topic to ConsumerFailurePolicy

        Raises:
            KeyError: If referenced policy key not found in configuration
        """
        failure_policies: Dict[str, ConsumerFailurePolicy] = {}

        # Guard against missing kafka consumer configuration to avoid attribute access on None
        if not getattr(config, "kafka_consumers", None):
            logger.info("No Kafka consumer subscriptions configured; skipping failure policy mapping")
            return failure_policies

        # Use a local resilience reference to avoid chained attribute access on None
        resilience = getattr(config.infrastructure, "resilience", None)
        if not (resilience and getattr(resilience, "consumer_failure_policies", None)):
            return failure_policies

        for sub in config.kafka_consumers.topic_subscriptions:
            if not sub.failure_policy_key:
                continue

            try:
                policy = resilience.get_consumer_failure_policy(sub.failure_policy_key)
                failure_policies[sub.topic] = policy
                logger.info(
                    f"Mapped topic '{sub.topic}' to failure policy '{sub.failure_policy_key}'",
                    extra={
                        "topic": sub.topic,
                        "max_retries": policy.max_retries,
                        "dlq_topic": policy.dlq_topic,
                    }
                )
            except KeyError as e:
                logger.error(
                    f"Failure policy '{sub.failure_policy_key}' not found in config: {e}"
                )
                raise

        return failure_policies

    def _create_dlq_producer_if_needed(
        self,
        config: PreprocessConfig,
        failure_policies: dict
    ) -> Optional[object]:
        """Create DLQ producer if any failure policy requires it.

        Args:
            config: Service configuration
            failure_policies: Mapping of topic to failure policy

        Returns:
            DLQ producer if needed, None otherwise
        """
        dlq_topics = {p.dlq_topic for p in failure_policies.values() if p.dlq_topic}
        if not dlq_topics:
            return None

        from drl_trading_common.adapter.messaging.kafka_producer_adapter import KafkaProducerAdapter
        from drl_trading_preprocess.application.config.resilience_constants import RETRY_CONFIG_KAFKA_DLQ

        logger.info(f"Creating DLQ producer for topics: {dlq_topics}")

        # Safely obtain retry config from resilience if present, otherwise use None
        resilience = getattr(config.infrastructure, "resilience", None)
        dlq_retry_config = None
        if resilience is not None:
            try:
                dlq_retry_config = resilience.get_retry_config(RETRY_CONFIG_KAFKA_DLQ)
            except Exception as e:
                logger.warning(
                    "Unable to obtain DLQ retry config from resilience configuration, proceeding without retry_config: %s",
                    e,
                    exc_info=True,
                )

        producer_config = config.infrastructure.kafka.get_producer_config()

        dlq_producer = KafkaProducerAdapter(
            producer_config=producer_config,
            retry_config=dlq_retry_config,
            dlq_topic=None,  # DLQ producer itself has no DLQ
        )

        logger.info("DLQ producer created successfully")
        return dlq_producer

    def _create_retry_producer_if_needed(
        self,
        config: PreprocessConfig,
        failure_policies: dict
    ) -> Optional[object]:
        """Create retry topic producer if any failure policy requires it.

        Args:
            config: Service configuration
            failure_policies: Mapping of topic to failure policy

        Returns:
            Retry producer if needed, None otherwise
        """
        retry_topics = {p.retry_topic for p in failure_policies.values() if p.retry_topic}
        if not retry_topics:
            return None

        from drl_trading_common.adapter.messaging.kafka_producer_adapter import KafkaProducerAdapter
        from drl_trading_preprocess.application.config.resilience_constants import RETRY_CONFIG_KAFKA_DLQ

        logger.info(f"Creating retry producer for topics: {retry_topics}")

        # Safely obtain retry config from resilience if present, otherwise use None
        resilience = getattr(config.infrastructure, "resilience", None)
        retry_config = None
        if resilience is not None:
            try:
                retry_config = resilience.get_retry_config(RETRY_CONFIG_KAFKA_DLQ)
            except Exception as e:
                logger.warning(
                    "Unable to obtain retry config from resilience configuration, proceeding without retry_config: %s",
                    e,
                    exc_info=True,
                )

        producer_config = config.infrastructure.kafka.get_producer_config()

        retry_producer = KafkaProducerAdapter(
            producer_config=producer_config,
            retry_config=retry_config,
            dlq_topic=None,  # Retry producer itself has no DLQ
        )

        logger.info("Retry producer created successfully")
        return retry_producer

    def _create_and_start_consumer(
        self,
        consumer_config: dict,
        topics: list,
        topic_handlers: dict,
        failure_policies: dict,
        dlq_producer: Optional[object],
        retry_producer: Optional[object],
        consumer_group_id: str,
        handler_count: int,
    ) -> None:
        """Create Kafka consumer and start it in background thread.

        Args:
            consumer_config: Kafka consumer configuration
            topics: List of topics to subscribe to
            topic_handlers: Mapping of topic to handler function
            failure_policies: Mapping of topic to failure policy
            dlq_producer: DLQ producer instance (optional)
            retry_producer: Retry topic producer instance (optional)
            consumer_group_id: Consumer group ID for logging
            handler_count: Number of registered handlers for logging
        """
        self._kafka_consumer = KafkaConsumerTopicAdapter(
            consumer_config=consumer_config,
            topics=topics,
            topic_handlers=topic_handlers,
            failure_policies=failure_policies if failure_policies else None,
            dlq_producer=dlq_producer,
            retry_producer=retry_producer,
        )

        self._kafka_consumer_thread = threading.Thread(
            target=self._kafka_consumer.start,
            daemon=False,
            name="kafka-consumer-preprocess"
        )
        self._kafka_consumer_thread.start()

        logger.info(
            "Kafka consumer started in background thread",
            extra={
                "group_id": consumer_group_id,
                "topics": topics,
                "handler_count": handler_count,
            }
        )

    def _run_main_loop(self) -> None:
        """
        Run the main service loop.

        FlaskServiceBootstrap will handle Flask server startup, but we could add
        additional background tasks here if needed.
        """
        super()._run_main_loop()


def bootstrap_preprocess_service() -> None:
    """
    Bootstrap the preprocess service using the standardized pattern.

    This function provides the standard bootstrap interface using
    the ServiceBootstrap framework.
    """
    bootstrap = PreprocessServiceBootstrap()
    bootstrap.start()


# Legacy alias for backward compatibility during transition
bootstrap_preprocess_service_standardized = bootstrap_preprocess_service
