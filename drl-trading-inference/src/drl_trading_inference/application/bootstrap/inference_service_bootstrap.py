"""
Inference service bootstrap using ServiceBootstrap framework.

Implements the standard service bootstrap pattern with Flask web interface
for health checks while maintaining the inference service's core functionality.
"""

import logging
from typing import List

from drl_trading_inference.application.di.inference_module import InferenceModule
from injector import Module

from drl_trading_common.infrastructure.bootstrap.flask_service_bootstrap import FlaskServiceBootstrap
from drl_trading_common.infrastructure.health.basic_health_checks import (
    SystemResourcesHealthCheck,
    ServiceStartupHealthCheck,
    ConfigurationHealthCheck,
)
from drl_trading_common.infrastructure.health.health_check import HealthCheck
from drl_trading_inference.application.config.inference_config import InferenceConfig

logger = logging.getLogger(__name__)


class InferenceServiceBootstrap(FlaskServiceBootstrap):
    """Bootstrap implementation for the inference service.

    Infrastructure responsibilities only:
    - Provide DI modules (wiring core inference ports/adapters)
    - Register health checks (system, startup, configuration)
    - Provide (future) route registrar if needed
    - Start/stop placeholder business logic hooks

    Domain logic (model loading, prediction pipelines) belongs to core layer
    services acquired via DI; keep this thin per hexagonal architecture.
    """

    def __init__(self) -> None:
        """Initialize the inference service bootstrap."""
        super().__init__(service_name="inference", config_class=InferenceConfig)
        self._startup_health_check = ServiceStartupHealthCheck("inference_startup")

    def get_dependency_modules(self, app_config: InferenceConfig) -> List[Module]:
        """Return DI modules using already-loaded config instance."""
        return [InferenceModule(app_config)]

    def get_health_checks(self) -> List[HealthCheck]:
        """Return health checks (always includes configuration check)."""
        return [
            SystemResourcesHealthCheck(
                name="inference_system_resources",
                cpu_threshold=85.0,
                memory_threshold=90.0,
            ),
            self._startup_health_check,
            ConfigurationHealthCheck(self.config, "inference_configuration"),  # type: ignore[arg-type]
        ]

    def _start_service(self) -> None:
        """Start inference business logic placeholder.

        Future: orchestration of model loading, prediction loop scheduling, etc.
        Currently: initialize placeholder components & mark startup healthy.
        """
        service_logger = logging.getLogger(__name__)
        try:
            service_logger.info("=== STARTING INFERENCE SERVICE BUSINESS LOGIC ===")
            self._initialize_inference_components()
            self._startup_health_check.mark_startup_completed(success=True)
            service_logger.info("=== INFERENCE SERVICE BUSINESS LOGIC INITIALIZED SUCCESSFULLY ===")
        except Exception as e:  # pragma: no cover - defensive path
            self._startup_health_check.mark_startup_completed(success=False, error_message=str(e))
            service_logger.error(f"Failed to start inference service: {e}", exc_info=True)
            raise

    def _stop_service(self) -> None:
        """Stop inference service-specific logic."""
        service_logger = logging.getLogger(__name__)
        service_logger.info("=== STOPPING INFERENCE SERVICE BUSINESS LOGIC ===")
        try:
            # Placeholder for future cleanup
            service_logger.info("Inference service business logic stopped successfully")
        except Exception as e:  # pragma: no cover
            service_logger.error(f"Error stopping inference service: {e}", exc_info=True)

    def _initialize_inference_components(self) -> None:
        """Initialize inference-specific components."""
        logger.info("Setting up inference components (placeholder)...")
        self._setup_model_loading()
        self._setup_predictions()
        self._setup_messaging()
        logger.info("Inference components initialized")

    def _setup_model_loading(self) -> None:
        """Setup model loading infrastructure."""
        logger.info("Setting up model loading (placeholder)...")
        # TODO: Implement model loading setup

    def _setup_predictions(self) -> None:
        """Setup prediction pipelines."""
        logger.info("Setting up prediction pipelines (placeholder)...")
        # TODO: Implement prediction setup

    def _setup_messaging(self) -> None:
        """Setup messaging infrastructure."""
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
