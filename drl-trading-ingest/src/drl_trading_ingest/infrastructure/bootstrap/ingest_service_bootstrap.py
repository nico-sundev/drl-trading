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
from drl_trading_common.infrastructure.health.health_check import HealthCheck
from drl_trading_ingest.adapter.web.ingest_route_registrar import IngestRouteRegistrar
from drl_trading_ingest.infrastructure.config.ingest_config import IngestConfig

logger = logging.getLogger(__name__)


class IngestServiceBootstrap(FlaskServiceBootstrap):
    """Bootstrap implementation for the ingest service.

    Responsibilities (infrastructure orchestration only):
    - Provide DI modules (hexagonal wiring)
    - Register health checks (system, startup, configuration)
    - Supply route registrar for Flask endpoints
    - Start/stop high-level business logic hooks (placeholder until real logic)

    Business/domain logic lives in core layer services wired through DI. This
    class must remain thin and infrastructure-focused per hexagonal principles.
    """

    def __init__(self) -> None:
        """Initialize the ingest service bootstrap."""
        super().__init__(service_name="drl-trading-ingest", config_class=IngestConfig)
        self._startup_health_check = ServiceStartupHealthCheck("ingest_startup")

    def get_dependency_modules(self, app_config: IngestConfig) -> List[Module]:
        """Return DI modules using already-loaded config instance.

        The config instance passed here is the single authoritative object
        produced during the config phase. We must NOT reload configuration
        inside DI modules (prevents divergence & duplicate IO). Import errors
        are surfaced immediately to fail fast.
        """
        from drl_trading_ingest.infrastructure.di.ingest_module import IngestModule  # type: ignore

        return [IngestModule(app_config)]

    def get_route_registrar(self) -> IngestRouteRegistrar:
        """Return ingest-specific route registrar for Flask endpoints."""
        return IngestRouteRegistrar()

    def get_health_checks(self) -> List[HealthCheck]:
        """Return health checks (always includes configuration check).

        Assumes configuration load phase already succeeded (else bootstrap
        would have aborted). Therefore we always add ConfigurationHealthCheck.
        """
        return [
            SystemResourcesHealthCheck(name="ingest_system_resources"),
            self._startup_health_check,
            ConfigurationHealthCheck(self.config, "ingest_configuration"),  # type: ignore[arg-type]
        ]

    def _start_service(self) -> None:
        """Start ingest service business logic placeholder.

        Real ingestion orchestration (e.g., background consumers, schedulers)
        will be triggered here once implemented. For now we just mark startup
        successful to feed health check status.
        """
        service_logger = logging.getLogger(__name__)
        try:
            service_logger.info("=== STARTING INGEST SERVICE BUSINESS LOGIC ===")
            service_logger.info("Initializing ingest service components (placeholder)...")
            service_logger.info("Dependency injection wiring completed")
            self._startup_health_check.mark_startup_completed(success=True)
            service_logger.info(
                "=== INGEST SERVICE BUSINESS LOGIC INITIALIZED SUCCESSFULLY ==="
            )
        except Exception as e:  # pragma: no cover - defensive guard
            self._startup_health_check.mark_startup_completed(success=False, error_message=str(e))
            service_logger.error(f"Failed to start ingest service: {e}", exc_info=True)
            raise

    def _stop_service(self) -> None:
        """Stop ingest service-specific logic with proper logging."""
        service_logger = logging.getLogger(__name__)
        service_logger.info("=== STOPPING INGEST SERVICE BUSINESS LOGIC ===")
        try:
            # Placeholder for future cleanup
            service_logger.info("Ingest service business logic stopped successfully")
        except Exception as e:  # pragma: no cover - defensive
            service_logger.error(f"Error stopping ingest service: {e}", exc_info=True)

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
