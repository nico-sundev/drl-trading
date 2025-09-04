"""
Preprocess service bootstrap using ServiceBootstrap framework.

Implements the standard service bootstrap pattern with Flask web interface
for health checks while maintaining the preprocessing service's core functionality.

Integrates the standardized T005 logging framework (ServiceLogger) so logs
include class/module names and service context consistently across services.
"""

import logging
from typing import List

from injector import Module

from drl_trading_adapter.infrastructure.di.adapter_module import AdapterModule
from drl_trading_common.infrastructure.bootstrap.flask_service_bootstrap import FlaskServiceBootstrap
from drl_trading_common.infrastructure.health.basic_health_checks import (
    SystemResourcesHealthCheck,
    ServiceStartupHealthCheck,
    ConfigurationHealthCheck,
)
from drl_trading_common.infrastructure.health.health_check import HealthCheck
from drl_trading_core.infrastructure.di.core_module import CoreModule
from drl_trading_preprocess.infrastructure.config.preprocess_config import PreprocessConfig


logger = logging.getLogger(__name__)


class PreprocessServiceBootstrap(FlaskServiceBootstrap):
    """Bootstrap implementation for the preprocess service.

    Responsibilities (infrastructure only):
    - Provide DI modules (fail fast on import issues)
    - Register health checks (system, startup, configuration)
    - Supply route registrar (default health-only endpoints for now)
    - Start/stop placeholder business logic hooks

    Domain logic (feature computation, data validation, etc.) will be delivered
    via DI-managed core services, keeping this class thin per hexagonal design.
    """

    def __init__(self) -> None:
        """Initialize the preprocess service bootstrap."""
        super().__init__(service_name="drl-trading-preprocess", config_class=PreprocessConfig)
        self._startup_health_check = ServiceStartupHealthCheck("preprocess_startup")

    def get_dependency_modules(self, app_config: PreprocessConfig) -> List[Module]:
        """Return DI modules using existing loaded config instance.

        The provided app_config avoids redundant config reloads inside DI.
        """
        from drl_trading_preprocess.infrastructure.di.preprocess_module import PreprocessModule  # type: ignore

        return [PreprocessModule(app_config), CoreModule(), AdapterModule()]

    def get_health_checks(self) -> List[HealthCheck]:
        """Return health checks (always includes configuration check)."""
        return [
            SystemResourcesHealthCheck(
                name="preprocess_system_resources",
                cpu_threshold=90.0,
                memory_threshold=90.0,
            ),
            self._startup_health_check,
            ConfigurationHealthCheck(self.config, "preprocess_configuration"),  # type: ignore[arg-type]
        ]

    # Logging is configured by the base ServiceBootstrap._setup_logging

    def _start_service(self) -> None:
        """Start preprocess business logic placeholder.

        Real preprocessing orchestration (feature pipelines, schedulers, etc.)
        will be hooked here. For now, we simply call component initialization
        placeholders and mark startup healthy for health reporting.
        """
        service_logger = logging.getLogger(__name__)
        try:
            service_logger.info("=== STARTING PREPROCESS SERVICE BUSINESS LOGIC ===")
            self._initialize_preprocessing_components()
            self._startup_health_check.mark_startup_completed(success=True)
            service_logger.info(
                "=== PREPROCESS SERVICE BUSINESS LOGIC INITIALIZED SUCCESSFULLY ==="
            )
        except Exception as e:  # pragma: no cover - defensive path
            self._startup_health_check.mark_startup_completed(success=False, error_message=str(e))
            service_logger.error("Failed to start preprocess service: %s", e, exc_info=True)
            raise

    def _stop_service(self) -> None:
        """Stop preprocess service-specific logic."""
        service_logger = logging.getLogger(__name__)
        service_logger.info("=== STOPPING PREPROCESS SERVICE BUSINESS LOGIC ===")
        try:
            # Placeholder for future cleanup
            service_logger.info("Preprocess service business logic stopped successfully")
        except Exception as e:  # pragma: no cover - defensive
            service_logger.error("Error stopping preprocess service: %s", e, exc_info=True)

    def _initialize_preprocessing_components(self) -> None:
        """Initialize preprocessing-specific components."""
        logger.info("Setting up preprocessing components...")
        self._setup_feature_computation()
        self._setup_data_validation()
        self._setup_feature_store()
        self._setup_messaging()
        logger.info("Preprocessing components initialized")

    def _setup_feature_computation(self) -> None:
        """Setup feature computation pipelines."""
        logger.info("Setting up feature computation (placeholder)...")
        # TODO: Implement feature computation setup

    def _setup_data_validation(self) -> None:
        """Setup data validation pipelines."""
        logger.info("Setting up data validation (placeholder)...")
        # TODO: Implement data validation setup

    def _setup_feature_store(self) -> None:
        """Setup feature store integration."""
        logger.info("Setting up feature store integration (placeholder)...")
        # TODO: Implement feature store setup

    def _setup_messaging(self) -> None:
        """Setup messaging infrastructure."""
        logger.info("Setting up messaging (placeholder)...")
        # TODO: Implement messaging setup

    def _run_main_loop(self) -> None:
        """
        Run the main service loop.

        FlaskServiceBootstrap will handle Flask server startup, but we could add
        additional background tasks here if needed.
        """
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
