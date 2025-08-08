"""
Preprocess service bootstrap using ServiceBootstrap framework.

Implements the standard service bootstrap pattern with Flask web interface
for health checks while maintaining the preprocessing service's core functionality.
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
from drl_trading_preprocess.infrastructure.config.preprocess_config import PreprocessConfig

logger = logging.getLogger(__name__)


class PreprocessServiceBootstrap(FlaskServiceBootstrap):
    """
    Bootstrap implementation for the preprocess service.

    Uses the specialized FlaskServiceBootstrap with automatic Flask web interface
    for health checks while running preprocessing workflows.
    """

    def __init__(self) -> None:
        """Initialize the preprocess service bootstrap."""
        super().__init__(service_name="preprocess", config_class=PreprocessConfig)
        self._startup_health_check = ServiceStartupHealthCheck("preprocess_startup")

    def get_dependency_modules(self) -> List[Module]:
        """
        Return dependency injection modules for this service.

        For now, returns empty list - preprocess service modules to be implemented.
        """
        # TODO: Implement preprocess service dependency injection modules
        logger.warning("Preprocess service dependency injection modules not yet implemented")
        return []

    def get_health_checks(self) -> List:
        """
        Return health checks for this service.

        Returns:
            List of HealthCheck instances for the preprocess service
        """
        health_checks = [
            SystemResourcesHealthCheck(
                name="preprocess_system_resources",
                cpu_threshold=90.0,  # Preprocessing can be CPU intensive
                memory_threshold=90.0  # Preprocessing can be memory intensive
            ),
            self._startup_health_check,
        ]

        # Add configuration health check if config is loaded
        if self.config:
            health_checks.append(ConfigurationHealthCheck(self.config, "preprocess_configuration"))

        return health_checks

    def _start_service(self) -> None:
        """
        Start preprocess service-specific logic.

        Initializes preprocessing workflows and core business services.
        """
        try:
            logger.info("Initializing preprocess service business logic...")

            # Mark startup as beginning
            self._startup_health_check.startup_completed = False

            # Initialize preprocessing components
            self._initialize_preprocessing_components()

            # Mark startup as completed successfully
            self._startup_health_check.mark_startup_completed(success=True)
            logger.info("Preprocess service business logic initialized successfully")

        except Exception as e:
            self._startup_health_check.mark_startup_completed(
                success=False,
                error_message=str(e)
            )
            logger.error(f"Failed to start preprocess service: {e}")
            raise

    def _stop_service(self) -> None:
        """Stop preprocess service-specific logic."""
        logger.info("Stopping preprocess service business logic...")
        try:
            # Any cleanup logic would go here
            logger.info("Preprocess service business logic stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping preprocess service: {e}")

    def _initialize_preprocessing_components(self) -> None:
        """Initialize preprocessing-specific components."""
        logger.info("Setting up preprocessing components...")

        # Setup feature computation
        self._setup_feature_computation()

        # Setup data validation
        self._setup_data_validation()

        # Setup feature store integration
        self._setup_feature_store()

        # Setup messaging
        self._setup_messaging()

        logger.info("Preprocessing components initialized")

    def _setup_feature_computation(self) -> None:
        """Setup feature computation pipelines."""
        logger.info("Setting up feature computation...")
        # TODO: Implement feature computation setup

    def _setup_data_validation(self) -> None:
        """Setup data validation pipelines."""
        logger.info("Setting up data validation...")
        # TODO: Implement data validation setup

    def _setup_feature_store(self) -> None:
        """Setup feature store integration."""
        logger.info("Setting up feature store integration...")
        # TODO: Implement feature store setup

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
