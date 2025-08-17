"""
Ingest service bootstrap using ServiceBootstrap framework.

Implements the standard service bootstrap pattern with Flask web interface
for health checks and data ingestion endpoints.
"""

import logging
from typing import List

from injector import Module

from drl_trading_common.infrastructure.bootstrap.flask_service_bootstrap import (
    FlaskServiceBootstrap,
)
from drl_trading_common.infrastructure.health.basic_health_checks import (
    ConfigurationHealthCheck,
    ServiceStartupHealthCheck,
    SystemResourcesHealthCheck,
)
from drl_trading_ingest.adapter.web.ingest_route_registrar import IngestRouteRegistrar
from drl_trading_ingest.infrastructure.config.ingest_config import IngestConfig

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
            health_checks.append(
                ConfigurationHealthCheck(self.config, "ingest_configuration")
            )

        return health_checks

    def _start_service(self) -> None:
        """Start ingest service logic (logging already configured)."""
        service_root_logger = logging.getLogger(__name__)
        try:
            service_root_logger.info("=== STARTING INGEST SERVICE BUSINESS LOGIC ===")
            self._startup_health_check.startup_completed = False
            service_root_logger.info("Initializing ingest service components...")
            service_root_logger.info("Dependency injection wiring completed")
            self._startup_health_check.mark_startup_completed(success=True)
            service_root_logger.info(
                "=== INGEST SERVICE BUSINESS LOGIC INITIALIZED SUCCESSFULLY ==="
            )
        except Exception as e:
            self._startup_health_check.mark_startup_completed(
                success=False, error_message=str(e)
            )
            service_root_logger.error(
                f"Failed to start ingest service: {e}", exc_info=True
            )
            raise

    def _stop_service(self) -> None:
        """Stop ingest service-specific logic with proper logging."""
        service_root_logger = logging.getLogger(self.service_name)
        service_root_logger.info("=== STOPPING INGEST SERVICE BUSINESS LOGIC ===")
        try:
            # Any cleanup logic would go here
            service_root_logger.info("Ingest service business logic stopped successfully")
        except Exception as e:
            service_root_logger.error(f"Error stopping ingest service: {e}", exc_info=True)

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
