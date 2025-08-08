"""
Inference service bootstrap using ServiceBootstrap framework.

Implements the standard service bootstrap pattern with Flask web interface
for health checks while maintaining the inference service's core functionality.
"""

import logging
from typing import List, cast

from drl_trading_inference.infrastructure.di.InferenceModule import InferenceModule
from injector import Module

from drl_trading_common.infrastructure.bootstrap.flask_service_bootstrap import FlaskServiceBootstrap
from drl_trading_common.infrastructure.health.basic_health_checks import (
    SystemResourcesHealthCheck,
    ServiceStartupHealthCheck,
    ConfigurationHealthCheck
)
from drl_trading_inference.infrastructure.config.inference_config import InferenceConfig

logger = logging.getLogger(__name__)


class InferenceServiceBootstrap(FlaskServiceBootstrap):
    """
    Bootstrap implementation for the inference service.

    Uses the specialized FlaskServiceBootstrap with automatic Flask web interface
    for health checks while running inference workflows.
    """

    def __init__(self) -> None:
        """Initialize the inference service bootstrap."""
        super().__init__(service_name="inference", config_class=InferenceConfig)
        self._startup_health_check = ServiceStartupHealthCheck("inference_startup")

    def get_dependency_modules(self) -> List[Module]:
        """
        Return dependency injection modules for this service.
        """
        typed_config = cast(InferenceConfig, self.config)
        return [InferenceModule(typed_config)]

    def get_health_checks(self) -> List:
        """
        Return health checks for this service.

        Returns:
            List of HealthCheck instances for the inference service
        """
        health_checks = [
            SystemResourcesHealthCheck(
                name="inference_system_resources",
                cpu_threshold=85.0,  # Inference should be responsive
                memory_threshold=90.0  # May need memory for models
            ),
            self._startup_health_check,
        ]

        # Add configuration health check if config is loaded
        if self.config:
            health_checks.append(ConfigurationHealthCheck(self.config, "inference_configuration"))

        return health_checks

    def _start_service(self) -> None:
        """
        Start inference service-specific logic.

        Initializes inference workflows and core business services.
        """
        try:
            logger.info("Initializing inference service business logic...")

            # Mark startup as beginning
            self._startup_health_check.startup_completed = False

            # Initialize inference components
            self._initialize_inference_components()

            # Mark startup as completed successfully
            self._startup_health_check.mark_startup_completed(success=True)
            logger.info("Inference service business logic initialized successfully")

        except Exception as e:
            self._startup_health_check.mark_startup_completed(
                success=False,
                error_message=str(e)
            )
            logger.error(f"Failed to start inference service: {e}")
            raise

    def _stop_service(self) -> None:
        """Stop inference service-specific logic."""
        logger.info("Stopping inference service business logic...")
        try:
            # Any cleanup logic would go here
            logger.info("Inference service business logic stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping inference service: {e}")

    def _initialize_inference_components(self) -> None:
        """Initialize inference-specific components."""
        logger.info("Setting up inference components...")

        # Setup model loading
        self._setup_model_loading()

        # Setup predictions
        self._setup_predictions()

        # Setup messaging
        self._setup_messaging()

        logger.info("Inference components initialized")

    def _setup_model_loading(self) -> None:
        """Setup model loading infrastructure."""
        logger.info("Setting up model loading...")
        # TODO: Implement model loading setup

    def _setup_predictions(self) -> None:
        """Setup prediction pipelines."""
        logger.info("Setting up prediction pipelines...")
        # TODO: Implement prediction setup

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


def bootstrap_inference_service() -> None:
    """
    Bootstrap the inference service using the standardized pattern.

    This function provides the standard bootstrap interface using
    the ServiceBootstrap framework.
    """
    bootstrap = InferenceServiceBootstrap()
    bootstrap.start()


# Legacy alias for backward compatibility during transition
bootstrap_inference_service_standardized = bootstrap_inference_service
