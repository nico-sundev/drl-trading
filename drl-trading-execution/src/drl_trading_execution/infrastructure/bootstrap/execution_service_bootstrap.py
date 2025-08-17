"""
Execution service bootstrap using ServiceBootstrap framework.

Implements the standard service bootstrap pattern with Flask web interface
for health checks while maintaining the execution service's core functionality.
"""

import logging
from typing import List

from injector import Module

from drl_trading_common.infrastructure.bootstrap.flask_service_bootstrap import FlaskServiceBootstrap
from drl_trading_common.infrastructure.health.basic_health_checks import (
    SystemResourcesHealthCheck,
    ServiceStartupHealthCheck,
    ConfigurationHealthCheck,
)
from drl_trading_common.infrastructure.health.health_check import HealthCheck
from drl_trading_execution.infrastructure.config.execution_config import ExecutionConfig
from drl_trading_execution.infrastructure.di.execution_module import ExecutionModule

logger = logging.getLogger(__name__)


class ExecutionServiceBootstrap(FlaskServiceBootstrap):
    """Bootstrap implementation for the execution service.

    Infrastructure responsibilities only:
    - Provide DI modules (future) wiring ports ↔ adapters
    - Register standard health checks (system, startup, configuration)
    - Expose web interface via Flask bootstrap
    - Start/stop placeholder business logic hooks

    Trading execution domain logic (order routing, risk checks) belongs in
    core services; keep this class thin per hexagonal architecture.
    """

    def __init__(self) -> None:
        """Initialize the execution service bootstrap."""
        super().__init__(service_name="execution", config_class=ExecutionConfig)
        self._startup_health_check = ServiceStartupHealthCheck("execution_startup")

    def get_dependency_modules(self, app_config: ExecutionConfig) -> List[Module]:
        """Return DI modules using already-loaded config instance.

        The config object is injected here to avoid redundant reloads inside
        module providers. Expand this list as adapters/ports are added.
        """
        return [ExecutionModule(app_config)]

    def get_health_checks(self) -> List[HealthCheck]:
        """Return health checks (always includes configuration check)."""
        return [
            SystemResourcesHealthCheck(
                name="execution_system_resources",
                cpu_threshold=80.0,
                memory_threshold=85.0,
            ),
            self._startup_health_check,
            ConfigurationHealthCheck(self.config, "execution_configuration"),  # type: ignore[arg-type]
        ]

    def _start_service(self) -> None:
        """Start execution business logic placeholder.

        Future: order routing engines, broker gateways, risk evaluators.
        Currently: initialize placeholder components & mark startup healthy.
        """
        service_logger = logging.getLogger(__name__)
        try:
            service_logger.info("=== STARTING EXECUTION SERVICE BUSINESS LOGIC ===")
            self._initialize_execution_components()
            self._startup_health_check.mark_startup_completed(success=True)
            service_logger.info(
                "=== EXECUTION SERVICE BUSINESS LOGIC INITIALIZED SUCCESSFULLY ==="
            )
        except Exception as e:  # pragma: no cover - defensive path
            self._startup_health_check.mark_startup_completed(success=False, error_message=str(e))
            service_logger.error(f"Failed to start execution service: {e}", exc_info=True)
            raise

    def _stop_service(self) -> None:
        """Stop execution service-specific logic."""
        service_logger = logging.getLogger(__name__)
        service_logger.info("=== STOPPING EXECUTION SERVICE BUSINESS LOGIC ===")
        try:
            # Placeholder for future cleanup
            service_logger.info("Execution service business logic stopped successfully")
        except Exception as e:  # pragma: no cover
            service_logger.error(f"Error stopping execution service: {e}", exc_info=True)

    def _initialize_execution_components(self) -> None:
        """Initialize execution-specific components (placeholders)."""
        logger.info("Setting up execution components (placeholder)...")
        self._setup_order_management()
        self._setup_risk_management()
        self._setup_broker_connections()
        self._setup_messaging()
        logger.info("Execution components initialized")

    def _setup_order_management(self) -> None:
        """Setup order management systems (placeholder)."""
        logger.info("Setting up order management (placeholder)...")
        # TODO: Implement order management setup

    def _setup_risk_management(self) -> None:
        """Setup risk management systems (placeholder)."""
        logger.info("Setting up risk management (placeholder)...")
        # TODO: Implement risk management setup

    def _setup_broker_connections(self) -> None:
        """Setup broker connection infrastructure (placeholder)."""
        logger.info("Setting up broker connections (placeholder)...")
        # TODO: Implement broker connection setup

    def _setup_messaging(self) -> None:
        """Setup messaging infrastructure (placeholder)."""
        logger.info("Setting up messaging (placeholder)...")
        # TODO: Implement messaging setup

    def _run_main_loop(self) -> None:
        """
        Run the main service loop.

        FlaskServiceBootstrap will handle Flask server startup,
        but we could add additional background tasks here if needed.
        """
        # FlaskServiceBootstrap handles Flask server in its _run_main_loop
        super()._run_main_loop()


def bootstrap_execution_service() -> None:
    """
    Bootstrap the execution service using the standardized pattern.

    This function provides the standard bootstrap interface using
    the ServiceBootstrap framework.
    """
    bootstrap = ExecutionServiceBootstrap()
    bootstrap.start()


# Legacy alias for backward compatibility during transition
bootstrap_execution_service_standardized = bootstrap_execution_service
