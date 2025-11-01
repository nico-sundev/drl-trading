"""
Service Bootstrap Framework for DRL Trading Services.

This module provides standardized bootstrap patterns for all deployable DRL Trading
microservices while maintaining strict hexagonal architecture compliance.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Type
import signal
import logging
import sys
import time
from drl_trading_common.infrastructure.web.generic_flask_app_factory import (
    RouteRegistrar,
)
from injector import Injector, Module
from flask import Flask

from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.logging.bootstrap_logging import retire_bootstrap_logger
from drl_trading_common.infrastructure.bootstrap.startup_context import StartupContext

logger = logging.getLogger(__name__)


class ServiceBootstrap(ABC):
    """
    Abstract base class for standardized service bootstrapping.

    HEXAGONAL ARCHITECTURE COMPLIANCE:
    - Belongs in infrastructure layer (technical concern)
    - Orchestrates dependency injection setup
    - Configures adapters but doesn't contain business logic
    - Core business services remain framework-agnostic

    Provides common patterns for:
    - Configuration loading
    - Dependency injection setup
    - Logging configuration
    - Graceful shutdown handling
    - Health check endpoints
    - Optional Flask web interface
    """

    def __init__(self, service_name: str, config_class: Type[BaseApplicationConfig]):
        """
        Initialize service bootstrap.

        Args:
            service_name: Name of the service (e.g., "inference", "training")
            config_class: Configuration class for this service
        """
        self.service_name = service_name
        self.config_class = config_class
        self.injector: Optional[Injector] = None
        self.is_running = False
        self._flask_app: Optional[Flask] = None
        self._setup_signal_handlers()

    def start(self, config_path: Optional[str] = None) -> None:
        """
        Start the service with standardized bootstrap sequence.

        HEXAGONAL ARCHITECTURE: This method orchestrates infrastructure setup
        but delegates business logic to core services via dependency injection.

        Bootstrap sequence:
        1. Load configuration
        2. Setup logging
        3. Initialize dependency injection (wire ports to adapters)
        4. Setup health checks
        5. Setup Flask web interface (if enabled)
        6. Start service-specific logic (via core services)

        Args:
            config_path: Optional path to configuration file
        """
        ctx = StartupContext(self.service_name)
        try:
            logger.info(f"Starting {self.service_name} service...")

            with ctx.phase("config"):
                self._load_configuration(config_path)
                stage = getattr(self.config, "stage", "unknown")
                ctx.attribute("stage", stage)
                ctx.attribute("has_logging_section", bool(getattr(self.config, "logging", None)))

            with ctx.phase("logging"):
                self._setup_logging()

            with ctx.phase("dependency_injection"):
                self._setup_dependency_injection()

            with ctx.phase("dependency_checks"):
                self._evaluate_startup_dependencies(ctx)

            with ctx.phase("health_checks"):
                self._setup_health_checks()

            with ctx.phase("web_interface"):
                self._setup_web_interface()
                ctx.attribute("web_interface_enabled", self.enable_web_interface())

            with ctx.phase("business_start"):
                self._start_service()

            self.is_running = True
            ctx.emit_summary(logger)
            if ctx.mandatory_dependencies_healthy():
                logger.info(f"{self.service_name} service started successfully")
            else:
                logger.warning(
                    f"{self.service_name} service started in DEGRADED state (see STARTUP SUMMARY)"
                )
            self._run_main_loop()
        except Exception as e:
            logger.error(f"Failed to start {self.service_name}: {e}")
            try:
                ctx.emit_summary(logger)
            except Exception:  # pragma: no cover
                pass
            self._cleanup()
            sys.exit(1)

    def stop(self) -> None:
        """Gracefully stop the service."""
        if self.is_running:
            logger.info(f"Stopping {self.service_name} service...")

            # Stop Kafka consumers first (if running)
            if hasattr(self, "_kafka_adapter") and self._kafka_adapter:
                logger.info("Stopping Kafka consumer adapter...")
                self._kafka_adapter.stop()

                # Wait for consumer thread to finish (with timeout)
                if hasattr(self, "_kafka_thread") and self._kafka_thread:
                    self._kafka_thread.join(timeout=10.0)
                    if self._kafka_thread.is_alive():
                        logger.warning("Kafka consumer thread did not stop within timeout")
                    else:
                        logger.info("Kafka consumer stopped successfully")

            # Stop service-specific logic
            self._stop_service()
            self._cleanup()
            self.is_running = False
            logger.info(f"{self.service_name} service stopped")

    def _load_configuration(self, config_path: Optional[str] = None) -> None:
        """
        Load service configuration using standardized loader.

        Uses the lean EnhancedServiceConfigLoader with secret substitution support.
        """
        from drl_trading_common.config.service_config_loader import ServiceConfigLoader
        # Delegates observability to ServiceConfigLoader (bootstrap logger used there)
        self.config = ServiceConfigLoader.load_config(self.config_class, service_name=self.service_name)

    def _setup_logging(self) -> None:
        """Setup standardized logging configuration."""
        try:
            from drl_trading_common.logging.service_logger import ServiceLogger

            # Retire bootstrap logger BEFORE full configuration to avoid duplicate handlers
            retire_bootstrap_logger(self.service_name)

            # Prefer top-level T005 logging config when present
            stage = getattr(self.config, "stage", "local") if self.config else "local"
            t005_cfg = getattr(self.config, "logging", None) if self.config else None
            ServiceLogger(
                service_name=self.service_name, stage=stage, config=t005_cfg
            ).configure()
        except Exception as e:  # pragma: no cover
            logger.warning(f"ServiceLogger configuration failed: {e}")

    def _setup_dependency_injection(self) -> None:
        """
        Initialize dependency injection container.

        HEXAGONAL ARCHITECTURE: This is where we wire ports to adapters:
        - Core services depend on port interfaces
        - Infrastructure modules bind ports to concrete adapters
        - Configuration flows from infrastructure to core via DI
        """
        # Pass the already-loaded configuration into DI module factory to ensure
        # a single authoritative config instance (avoid each Module re-loading).
        di_modules = self.get_dependency_modules(self.config)  # type: ignore[arg-type]
        self.injector = Injector(di_modules)
        logger.info(
            "Dependency injection container initialized (hexagonal architecture wired)"
        )

    def _evaluate_startup_dependencies(self, ctx: StartupContext) -> None:
        """Evaluate critical external dependencies and record health.

        Currently checks:
          - Database connectivity (if DatabaseConnectionInterface is bound)
        Additional checks (messaging, feature store, etc.) can be added here.
        """
        if self.injector is None:
            return
        try:
            from drl_trading_common.db.database_connection_interface import (
                DatabaseConnectionInterface,  # type: ignore
                DatabaseConnectionError,
            )

            start = time.time()
            try:
                db_service = self.injector.get(DatabaseConnectionInterface)  # type: ignore
            except Exception:
                ctx.add_dependency_status(
                    name="database",
                    healthy=True,
                    mandatory=False,
                    message="No database binding (skipped)",
                )
                db_service = None
            if db_service:
                try:
                    with db_service.get_connection():  # type: ignore
                        latency = (time.time() - start) * 1000
                        ctx.add_dependency_status(
                            name="database",
                            healthy=True,
                            mandatory=True,
                            message="Connection pool healthy",
                            latency_ms=round(latency, 2),
                        )
                except DatabaseConnectionError as de:  # type: ignore
                    latency = (time.time() - start) * 1000
                    ctx.add_dependency_status(
                        name="database",
                        healthy=False,
                        mandatory=True,
                        message=str(de),
                        latency_ms=round(latency, 2),
                    )
        except Exception as e:  # pragma: no cover
            logger.debug(f"Startup dependency evaluation failed: {e}")

    def _setup_health_checks(self) -> None:
        """Setup standardized health check endpoints."""
        try:
            from drl_trading_common.infrastructure.health.health_check_service import (
                HealthCheckService,
            )

            if self.injector is None:
                logger.warning(
                    "Dependency injector is not initialized; skipping health check setup."
                )
                return
            health_service = self.injector.get(HealthCheckService)
            health_service.register_checks(self.get_health_checks())
            logger.info("Health checks configured")
        except Exception as e:
            logger.warning(f"Health check setup failed: {e}")

    def _setup_web_interface(self) -> None:
        """Setup Flask web interface if enabled."""
        if self.enable_web_interface():
            try:
                from drl_trading_common.infrastructure.web.generic_flask_app_factory import (
                    GenericFlaskAppFactory,
                    DefaultRouteRegistrar,
                )

                if self.injector is None:
                    raise RuntimeError(
                        "Injector must be initialized before setting up web interface."
                    )

                route_registrar = self.get_route_registrar() or DefaultRouteRegistrar()
                self._flask_app = GenericFlaskAppFactory.create_app(
                    service_name=self.service_name,
                    injector=self.injector,
                    config=self.config,
                    route_registrar=route_registrar,
                )
                logger.info(f"Flask web interface configured for {self.service_name}")
            except Exception as e:
                logger.error(f"Failed to setup web interface: {e}")
                raise

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown signal handlers."""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum: int, frame: object) -> None:
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()

    def _start_kafka_consumers(self) -> None:
        """
        Initialize and start Kafka consumers based on service configuration.

        This method:
        1. Checks if Kafka is configured in infrastructure config
        2. Retrieves handler registry from DI container
        3. Maps topics to handlers based on YAML configuration
        4. Creates and starts KafkaConsumerAdapter in a background thread

        Design: Configuration-driven, DI-injected handler mapping
        - Topic names come from YAML (infrastructure.kafka_consumers.topic_subscriptions)
        - Handler implementations come from DI (provide_kafka_handler_registry)
        - No hardcoded topics or handlers in bootstrap code

        Should be called from service-specific bootstrap's _run_main_loop()
        or _start_service() method.

        Thread Safety: Starts consumer in a daemon=False thread for graceful shutdown.
        """
        import threading
        from typing import Dict
        from drl_trading_common.messaging.kafka_message_handler import KafkaMessageHandler
        from drl_trading_common.adapter.messaging.kafka_consumer_adapter import KafkaConsumerAdapter

        # Check if Kafka is configured
        infrastructure = getattr(self.config, "infrastructure", None)
        if not infrastructure:
            logger.debug("No infrastructure config found, skipping Kafka consumers")
            return

        kafka_config = getattr(infrastructure, "kafka", None)
        kafka_consumer_config = getattr(self.config, "kafka_consumers", None)

        if not kafka_config or not kafka_consumer_config:
            logger.debug("No Kafka consumer configuration found, skipping")
            return

        # Get handler registry from DI container
        if not self.injector:
            logger.warning("DI injector not initialized, cannot start Kafka consumers")
            return

        try:
            from drl_trading_common.messaging.kafka_handler_registry import KafkaHandlerRegistry
            handler_registry: KafkaHandlerRegistry = self.injector.get(KafkaHandlerRegistry)  # type: ignore
        except Exception as e:
            logger.warning(
                f"No Kafka handler registry found in DI container: {e}. "
                "If this service needs Kafka consumers, ensure provide_kafka_handler_registry() "
                "is implemented in the DI module."
            )
            return

        # Build topic-to-handler mapping from config
        topic_handlers: Dict[str, KafkaMessageHandler] = {}
        topic_subscriptions = getattr(kafka_consumer_config, "topic_subscriptions", [])

        for subscription in topic_subscriptions:
            topic = subscription.topic
            handler_id = subscription.handler_id

            handler = handler_registry.get_handler(handler_id)
            if not handler:
                logger.warning(
                    f"No handler found for handler_id '{handler_id}' (topic: '{topic}'). "
                    f"Available handlers: {handler_registry.list_handler_ids()}"
                )
                continue

            topic_handlers[topic] = handler
            logger.info(
                "Registered Kafka handler",
                extra={
                    "topic": topic,
                    "handler_id": handler_id,
                    "service": self.service_name
                }
            )

        if not topic_handlers:
            logger.warning("No valid topic handlers configured, skipping Kafka consumer startup")
            return

        # Build consumer config
        consumer_group_id = getattr(kafka_consumer_config, "consumer_group_id", None)
        if not consumer_group_id:
            logger.error("consumer_group_id not configured, cannot start Kafka consumer")
            return

        consumer_config = kafka_config.get_consumer_config(group_id=consumer_group_id)

        # Create adapter
        self._kafka_adapter = KafkaConsumerAdapter(
            consumer_config=consumer_config,
            topic_handlers=topic_handlers
        )

        # Start in background thread (not daemon - we want graceful shutdown)
        self._kafka_thread = threading.Thread(
            target=self._kafka_adapter.start,
            daemon=False,
            name=f"kafka-consumer-{self.service_name}"
        )
        self._kafka_thread.start()

        logger.info(
            "Kafka consumer started in background thread",
            extra={
                "service": self.service_name,
                "group_id": consumer_group_id,
                "topics": list(topic_handlers.keys()),
                "handler_count": len(topic_handlers)
            }
        )

    # Abstract methods for service-specific implementation
    @abstractmethod
    def get_dependency_modules(self, app_config: BaseApplicationConfig) -> List[Module]:
        """
        Return dependency injection modules for this service.

        HEXAGONAL ARCHITECTURE: Should return modules that:
        - Bind core port interfaces to adapter implementations
        - Configure infrastructure concerns (logging, messaging, etc.)
        - Keep core business logic framework-agnostic

        Example:
        return [
            CoreModule(app_config),      # Core business services
            AdapterModule(app_config),   # External adapters (web, messaging)
            InfrastructureModule(app_config)  # Technical concerns
        ]
        """
        pass

    @abstractmethod
    def _start_service(self) -> None:
        """
        Start service-specific logic.

        HEXAGONAL ARCHITECTURE: Should delegate to core services via DI:
        - Get core application service from injector
        - Start business logic (not infrastructure concerns)
        - Let adapters handle external communication
        """
        pass

    @abstractmethod
    def _stop_service(self) -> None:
        """
        Stop service-specific logic.

        Should cleanly shutdown core business logic.
        """
        pass

    @abstractmethod
    def _run_main_loop(self) -> None:
        """
        Run the main service loop.

        For message-driven services, this might just wait.
        For request-response services, this might start the web server.
        For Flask services, this should start the Flask server if web interface is enabled.
        """
        pass

    def enable_web_interface(self) -> bool:
        """
        Override to enable Flask web interface for this service.

        Returns:
            True if service should expose Flask endpoints, False otherwise
        """
        return False

    def get_route_registrar(self) -> Optional[RouteRegistrar]:
        """
        Return service-specific route registrar for Flask endpoints.

        Returns:
            RouteRegistrar instance or None for default health-only endpoints
        """
        return None

    def get_health_checks(self) -> List:
        """
        Return health checks for this service (optional override).

        Returns:
            List of HealthCheck instances for this service
        """
        return []

    def get_flask_app(self) -> Optional[Flask]:
        """Get the Flask application instance if web interface is enabled."""
        return self._flask_app

    def _cleanup(self) -> None:
        """Cleanup resources before shutdown."""
        pass
