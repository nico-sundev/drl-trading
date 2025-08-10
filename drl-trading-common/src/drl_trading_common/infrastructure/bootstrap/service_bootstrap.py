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
from drl_trading_common.infrastructure.web.generic_flask_app_factory import (
    RouteRegistrar,
)
from injector import Injector, Module

from drl_trading_common.base.base_application_config import BaseApplicationConfig

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
        self.config: Optional[BaseApplicationConfig] = None
        self.injector: Optional[Injector] = None
        self.is_running = False
        self._flask_app = None
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
        try:
            logger.info(f"Starting {self.service_name} service...")

            # Step 1: Load configuration (infrastructure concern)
            self._load_configuration(config_path)

            # Step 2: Setup logging (infrastructure concern)
            self._setup_logging()

            # Step 3: Initialize dependency injection (wire hexagonal architecture)
            self._setup_dependency_injection()

            # Step 4: Setup health checks (infrastructure concern)
            self._setup_health_checks()

            # Step 5: Setup Flask web interface if enabled (infrastructure concern)
            self._setup_web_interface()

            # Step 6: Start service-specific logic (delegate to core via DI)
            self._start_service()

            self.is_running = True
            logger.info(f"{self.service_name} service started successfully")

            # Keep service running
            self._run_main_loop()

        except Exception as e:
            logger.error(f"Failed to start {self.service_name}: {e}")
            self._cleanup()
            sys.exit(1)

    def stop(self) -> None:
        """Gracefully stop the service."""
        if self.is_running:
            logger.info(f"Stopping {self.service_name} service...")
            self._stop_service()
            self._cleanup()
            self.is_running = False
            logger.info(f"{self.service_name} service stopped")

    def _load_configuration(self, config_path: Optional[str] = None) -> None:
        """
        Load service configuration using standardized loader.

        Uses the lean EnhancedServiceConfigLoader with secret substitution support.
        """
        from drl_trading_common.config.service_config_loader import (
            ServiceConfigLoader,
        )

        self.config = ServiceConfigLoader.load_config(self.config_class)
        logger.info(f"Configuration loaded for {self.service_name}")

    def _setup_logging(self) -> None:
        """Setup standardized logging configuration."""
        try:
            from drl_trading_common.logging.service_logger import ServiceLogger
            # Prefer top-level T005 logging config when present
            stage = getattr(self.config, "stage", "local") if self.config else "local"
            t005_cfg = getattr(self.config, "logging", None) if self.config else None
            ServiceLogger(service_name=self.service_name, stage=stage, config=t005_cfg).configure()
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
        di_modules = self.get_dependency_modules()
        self.injector = Injector(di_modules)
        logger.info(
            "Dependency injection container initialized (hexagonal architecture wired)"
        )

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

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()

    # Abstract methods for service-specific implementation
    @abstractmethod
    def get_dependency_modules(self) -> List[Module]:
        """
        Return dependency injection modules for this service.

        HEXAGONAL ARCHITECTURE: Should return modules that:
        - Bind core port interfaces to adapter implementations
        - Configure infrastructure concerns (logging, messaging, etc.)
        - Keep core business logic framework-agnostic

        Example:
        return [
            CoreModule(self.config),      # Core business services
            AdapterModule(self.config),   # External adapters (web, messaging)
            InfrastructureModule(self.config)  # Technical concerns
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

    def get_flask_app(self):
        """Get the Flask application instance if web interface is enabled."""
        return self._flask_app

    def _cleanup(self) -> None:
        """Cleanup resources before shutdown."""
        pass
