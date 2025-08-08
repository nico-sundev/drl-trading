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
    ConfigurationHealthCheck
)
from drl_trading_execution.infrastructure.config.execution_config import ExecutionConfig

logger = logging.getLogger(__name__)


class ExecutionServiceBootstrap(FlaskServiceBootstrap):
    """
    Bootstrap implementation for the execution service.

    Uses the specialized FlaskServiceBootstrap with automatic Flask web interface
    for health checks while running execution workflows.
    """

    def __init__(self) -> None:
        """Initialize the execution service bootstrap."""
        super().__init__(service_name="execution", config_class=ExecutionConfig)
        self._startup_health_check = ServiceStartupHealthCheck("execution_startup")

    def get_dependency_modules(self) -> List[Module]:
        """
        Return dependency injection modules for this service.

        For now, returns empty list - execution service modules to be implemented.
        """
        # TODO: Implement execution service dependency injection modules
        logger.warning("Execution service dependency injection modules not yet implemented")
        return []

    def get_health_checks(self) -> List:
        """
        Return health checks for this service.

        Returns:
            List of HealthCheck instances for the execution service
        """
        health_checks = [
            SystemResourcesHealthCheck(
                name="execution_system_resources",
                cpu_threshold=80.0,  # Execution should be highly responsive
                memory_threshold=85.0  # Should maintain low resource usage
            ),
            self._startup_health_check,
        ]

        # Add configuration health check if config is loaded
        if self.config:
            health_checks.append(ConfigurationHealthCheck(self.config, "execution_configuration"))

        return health_checks

    def _start_service(self) -> None:
        """
        Start execution service-specific logic.

        Initializes execution workflows and core business services.
        """
        try:
            logger.info("Initializing execution service business logic...")

            # Mark startup as beginning
            self._startup_health_check.startup_completed = False

            # Initialize execution components
            self._initialize_execution_components()

            # Mark startup as completed successfully
            self._startup_health_check.mark_startup_completed(success=True)
            logger.info("Execution service business logic initialized successfully")

        except Exception as e:
            self._startup_health_check.mark_startup_completed(
                success=False,
                error_message=str(e)
            )
            logger.error(f"Failed to start execution service: {e}")
            raise

    def _stop_service(self) -> None:
        """Stop execution service-specific logic."""
        logger.info("Stopping execution service business logic...")
        try:
            # Any cleanup logic would go here
            logger.info("Execution service business logic stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping execution service: {e}")

    def _initialize_execution_components(self) -> None:
        """Initialize execution-specific components."""
        logger.info("Setting up execution components...")

        # Setup order management
        self._setup_order_management()

        # Setup risk management
        self._setup_risk_management()

        # Setup broker connections
        self._setup_broker_connections()

        # Setup messaging
        self._setup_messaging()

        logger.info("Execution components initialized")

    def _setup_order_management(self) -> None:
        """Setup order management systems."""
        logger.info("Setting up order management...")
        # TODO: Implement order management setup

    def _setup_risk_management(self) -> None:
        """Setup risk management systems."""
        logger.info("Setting up risk management...")
        # TODO: Implement risk management setup

    def _setup_broker_connections(self) -> None:
        """Setup broker connection infrastructure."""
        logger.info("Setting up broker connections...")
        # TODO: Implement broker connection setup

    def _setup_messaging(self) -> None:
        """Setup messaging infrastructure."""
        logger.info("Setting up messaging...")
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
