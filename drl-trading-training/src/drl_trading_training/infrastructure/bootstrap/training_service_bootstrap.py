"""
Training service bootstrap for batch/job execution.

Implements the ServiceBootstrap pattern for batch processing without web interface
since training is a finite job, not a long-running service.
"""

import logging
from typing import List

from injector import Module

from drl_trading_common.infrastructure.bootstrap.service_bootstrap import ServiceBootstrap
from drl_trading_common.infrastructure.health.basic_health_checks import (
    SystemResourcesHealthCheck,
    ServiceStartupHealthCheck,
    ConfigurationHealthCheck
)
from drl_trading_training.infrastructure.config.training_config import TrainingConfig

logger = logging.getLogger(__name__)


class TrainingServiceBootstrap(ServiceBootstrap):
    """
    Bootstrap implementation for the training batch service.

    Uses the base ServiceBootstrap class since training is a finite job
    that doesn't require a web interface (Flask) like long-running services.
    """

    def __init__(self) -> None:
        """Initialize the training service bootstrap."""
        super().__init__(service_name="training", config_class=TrainingConfig)
        self._startup_health_check = ServiceStartupHealthCheck("training_startup")

    def get_dependency_modules(self) -> List[Module]:
        """
        Return dependency injection modules for this service.

        For now, returns empty list - training service modules to be implemented.
        """
        # TODO: Implement training service dependency injection modules
        logger.warning("Training service dependency injection modules not yet implemented")
        return []

    def get_health_checks(self) -> List:
        """
        Return health checks for this service.

        Returns:
            List of HealthCheck instances for the training service
        """
        health_checks = [
            SystemResourcesHealthCheck(
                name="training_system_resources",
                cpu_threshold=95.0,  # Training can be CPU intensive
                memory_threshold=95.0  # Training can be memory intensive
            ),
            self._startup_health_check,
        ]

        # Add configuration health check if config is loaded
        if self.config:
            health_checks.append(ConfigurationHealthCheck(self.config, "training_configuration"))

        return health_checks

    def _start_service(self) -> None:
        """
        Start training service-specific logic.

        Since training is a batch job, this initializes training components.
        """
        try:
            logger.info("Initializing training service business logic...")

            # Mark startup as beginning
            self._startup_health_check.startup_completed = False

            # Initialize training components
            self._initialize_training_components()

            # Mark startup as completed successfully
            self._startup_health_check.mark_startup_completed(success=True)
            logger.info("Training service business logic initialized successfully")

        except Exception as e:
            self._startup_health_check.mark_startup_completed(
                success=False,
                error_message=str(e)
            )
            logger.error(f"Failed to start training service: {e}")
            raise

    def _stop_service(self) -> None:
        """Stop training service-specific logic."""
        logger.info("Stopping training service business logic...")
        try:
            # Any cleanup logic would go here
            logger.info("Training service business logic stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping training service: {e}")

    def _run_main_loop(self) -> None:
        """
        Run the main training batch job.

        For batch services like training, we execute the job and then exit.
        This is different from long-running services that keep a loop alive.
        """
        try:
            # Execute the training workflow
            self._execute_training_workflow()

            logger.info("Training batch job completed successfully")
            # Training jobs should exit after completion
            self.stop()

        except Exception as e:
            logger.error(f"Training batch job failed: {e}")
            self.stop()
            raise
    def _execute_training_workflow(self) -> None:
        """Execute the main training workflow logic."""
        logger.info("Executing training workflow...")

        # Setup experiment tracking
        self._setup_experiment_tracking()

        # Setup datasets
        self._setup_datasets()

        # Setup agents
        self._setup_agents()

        # Setup messaging
        self._setup_messaging()

        # Run training
        self._run_training_jobs()

        logger.info("Training workflow execution completed")

    def _initialize_training_components(self) -> None:
        """Initialize training-specific components."""
        logger.info("Setting up training components...")

        # TODO: Component initialization will be implemented here

        logger.info("Training components initialized")

    def _setup_experiment_tracking(self) -> None:
        """Setup experiment tracking infrastructure."""
        logger.info("Setting up experiment tracking...")
        # TODO: Implement experiment tracking setup

    def _setup_datasets(self) -> None:
        """Setup dataset management."""
        logger.info("Setting up datasets...")
        # TODO: Implement dataset setup

    def _setup_agents(self) -> None:
        """Setup training agents."""
        logger.info("Setting up training agents...")
        # TODO: Implement agent setup

    def _setup_messaging(self) -> None:
        """Setup messaging infrastructure."""
        logger.info("Setting up messaging...")
        # TODO: Implement messaging setup

    def _run_training_jobs(self) -> None:
        """Execute training jobs."""
        logger.info("Running training jobs...")
        # TODO: Implement actual training execution


def bootstrap_training_service() -> None:
    """
    Bootstrap the training service as a batch job.

    Training service runs as a finite batch job, not a long-running service.
    """
    bootstrap = TrainingServiceBootstrap()
    try:
        # Start the service - this will initialize and run the training job
        bootstrap.start()

    except KeyboardInterrupt:
        logger.info("Training service interrupted by user")
    except Exception as e:
        logger.error(f"Training service failed: {e}")
        raise


# Maintain backward compatibility
bootstrap_training_service_standardized = bootstrap_training_service
