"""
Ingest service bootstrap using ServiceBootstrap framework.

Implements the standard service bootstrap pattern with Flask web interface
for health checks and data ingestion endpoints.
"""

import logging
from typing import List, Optional

from injector import Module

from drl_trading_common.infrastructure.bootstrap.flask_service_bootstrap import FlaskServiceBootstrap
from drl_trading_common.infrastructure.health.basic_health_checks import (
    SystemResourcesHealthCheck,
    ServiceStartupHealthCheck,
    ConfigurationHealthCheck
)
from drl_trading_common.logging.service_logger import ServiceLogger
from drl_trading_ingest.infrastructure.config.ingest_config import IngestConfig
from drl_trading_ingest.adapter.web.ingest_route_registrar import IngestRouteRegistrar

logger = logging.getLogger(__name__)


class IngestServiceBootstrap(FlaskServiceBootstrap):
    """
    Bootstrap implementation for the ingest service.

    Uses the specialized FlaskServiceBootstrap with automatic Flask web interface
    for health checks and data ingestion endpoints. Integrates standardized
    logging with trading context tracking.
    """

    def __init__(self) -> None:
        """Initialize the ingest service bootstrap."""
        super().__init__(service_name="drl-trading-ingest", config_class=IngestConfig)
        self._startup_health_check = ServiceStartupHealthCheck("ingest_startup")
        # Using forward reference since ServiceLogger is imported after this line
        self._service_logger: Optional[ServiceLogger] = None

    def get_dependency_modules(self) -> List[Module]:
        """
        Return dependency injection modules for this service.

        Uses the existing IngestModule which properly wires the hexagonal architecture.
        """
        try:
            from drl_trading_ingest.infrastructure.di.ingest_module import IngestModule
            return [IngestModule()]
        except ImportError as e:
            logger.error(f"Failed to import IngestModule: {e}")
            return []

    def _setup_standardized_logging(self) -> None:
        """Setup standardized logging for the ingest service."""
        class_logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        if self.config and hasattr(self.config, 'logging'):
            # Use T005 ServiceLogger with configuration
            self._service_logger = ServiceLogger(
                service_name="drl-trading-ingest",
                stage=self.config.stage,
                config=self.config.logging
            )
            self._service_logger.configure()

            # Get the configured logger and log setup success
            class_logger.info("ServiceLogger configured with full configuration from config file")
        else:
            # Fallback to basic ServiceLogger
            self._service_logger = ServiceLogger("drl-trading-ingest", self.config.stage if self.config else "local")
            self._service_logger.configure()

            class_logger.warning("ServiceLogger configured with defaults (no logging config found)")

    def get_service_logger(self) -> Optional[ServiceLogger]:
        """Get the configured ServiceLogger instance."""
        return self._service_logger

    def get_route_registrar(self) -> IngestRouteRegistrar:
        """Return ingest-specific route registrar for Flask endpoints."""
        return IngestRouteRegistrar()

    def get_health_checks(self) -> List:
        """
        Return health checks for this service.

        Returns:
            List of HealthCheck instances for the ingest service
        """
        health_checks = [
            SystemResourcesHealthCheck(name="ingest_system_resources"),
            self._startup_health_check,
        ]

        # Add configuration health check if config is loaded
        if self.config:
            health_checks.append(ConfigurationHealthCheck(self.config, "ingest_configuration"))

        return health_checks

    def _start_service(self) -> None:
        """
        Start ingest service-specific logic.

        Initializes core business services via dependency injection and
        sets up standardized logging with trading context tracking.
        """
        # Use class-specific logger for better traceability
        class_logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        try:
            class_logger.info("=== STARTING INGEST SERVICE BUSINESS LOGIC ===")

            # Setup standardized logging first
            self._setup_standardized_logging()

            # Log after logging setup to verify it's working
            class_logger.info("Standardized logging configured for ingest service")

            # Mark startup as beginning
            self._startup_health_check.startup_completed = False
            class_logger.info("Initializing ingest service components...")

            # Any additional service-specific initialization would go here
            # Core business logic is handled via dependency injection
            class_logger.info("Dependency injection wiring completed")

            # Mark startup as completed successfully
            self._startup_health_check.mark_startup_completed(success=True)
            class_logger.info("=== INGEST SERVICE BUSINESS LOGIC INITIALIZED SUCCESSFULLY ===")

        except Exception as e:
            self._startup_health_check.mark_startup_completed(
                success=False,
                error_message=str(e)
            )
            class_logger.error(f"Failed to start ingest service: {e}", exc_info=True)
            raise

    def _stop_service(self) -> None:
        """Stop ingest service-specific logic with proper logging."""
        class_logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        class_logger.info("=== STOPPING INGEST SERVICE BUSINESS LOGIC ===")
        try:
            # Any cleanup logic would go here
            class_logger.info("Ingest service business logic stopped successfully")
        except Exception as e:
            class_logger.error(f"Error stopping ingest service: {e}", exc_info=True)

    def _run_main_loop(self) -> None:
        """
        Run the main service loop.

        FlaskServiceBootstrap will handle Flask server startup,
        but we could add additional background tasks here if needed.
        """
        # FlaskServiceBootstrap handles Flask server in its _run_main_loop
        super()._run_main_loop()


def bootstrap_ingest_service() -> None:
    """
    Bootstrap the ingest service using the standardized pattern.

    This function provides the standard bootstrap interface using
    the ServiceBootstrap framework.
    """
    bootstrap = IngestServiceBootstrap()
    bootstrap.start()


# Legacy alias for backward compatibility during transition
bootstrap_ingest_service_standardized = bootstrap_ingest_service
