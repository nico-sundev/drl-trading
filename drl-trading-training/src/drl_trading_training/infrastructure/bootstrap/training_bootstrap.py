"""Bootstrap implementation for training service following T004 patterns."""
import logging
from typing import Optional

from drl_trading_common.config.logging_config import configure_unified_logging
from drl_trading_common.config.enhanced_service_config_loader import EnhancedServiceConfigLoader

# Use relative import to avoid module path issues
from ...infrastructure.config.training_config import TrainingConfig

# Configure basic logging for bootstrap phase
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def setup_logging(config: Optional[TrainingConfig] = None) -> None:
    """Set up logging using the unified configuration approach.

    Args:
        config: Optional TrainingConfig instance with logging settings
    """
    configure_unified_logging(config, service_name="training")
    logger.info("Logging configured using unified configuration approach")


def bootstrap_training_service() -> None:
    """Bootstrap the training service with proper configuration."""
    try:
        # Load configuration using lean EnhancedServiceConfigLoader
        logger.info("Loading configuration with lean EnhancedServiceConfigLoader")
        config = EnhancedServiceConfigLoader.load_config(TrainingConfig)

        # Reconfigure logging with loaded config
        setup_logging(config)

        # Log effective configuration for debugging
        logger.info(
            f"Training service initialized in {config.stage} mode "
            f"for {config.app_name} v{config.version}"
        )

        # Configure service components based on the config
        setup_experiment_tracking(config)
        setup_datasets(config)
        setup_agents(config)
        setup_messaging(config)
        setup_monitoring(config)

        # Start training
        start_training_loop(config)

    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        exit(1)
    except Exception as e:
        logger.exception(f"Failed to initialize training service: {e}")
        exit(2)


def setup_experiment_tracking(config: TrainingConfig) -> None:
    """Set up experiment tracking with MLflow."""
    if config.experiment_tracking.enabled:
        experiment_name = config.experiment_tracking.mlflow.experiment_name
        tracking_uri = config.experiment_tracking.mlflow.tracking_uri

        logger.info(
            f"Setting up MLflow experiment tracking: '{experiment_name}' "
            f"(URI: {tracking_uri or 'default'})"
        )
    else:
        logger.info("Experiment tracking disabled")
    # ... experiment tracking setup logic here


def setup_datasets(config: TrainingConfig) -> None:
    """Set up dataset loading and preprocessing."""
    input_path = config.dataset.input_path
    symbols = config.dataset.symbols
    train_split = config.dataset.train_split

    logger.info(
        f"Setting up datasets from {input_path} "
        f"(symbols: {symbols}, train_split: {train_split})"
    )
    # ... dataset setup logic here


def setup_agents(config: TrainingConfig) -> None:
    """Set up RL agents for training."""
    algorithms = config.agent.algorithms
    learning_rate = config.agent.hyperparameters.get("learning_rate", 0.0003)

    logger.info(
        f"Setting up RL agents: {algorithms} "
        f"(learning_rate: {learning_rate})"
    )
    # ... agent setup logic here


def setup_messaging(config: TrainingConfig) -> None:
    """Set up message bus connections."""
    provider = config.infrastructure.messaging.provider

    logger.info(f"Setting up messaging with {provider} provider")
    # ... messaging setup logic here


def setup_monitoring(config: TrainingConfig) -> None:
    """Set up monitoring and health checks."""
    logger.info("Setting up monitoring and health checks")
    # ... monitoring setup logic here


def start_training_loop(config: TrainingConfig) -> None:
    """Start the main training loop."""
    total_timesteps = config.experiment_tracking.hyperparameters.total_timesteps
    eval_frequency = config.experiment_tracking.hyperparameters.eval_frequency

    logger.info(
        f"Starting training loop "
        f"(timesteps: {total_timesteps}, eval_frequency: {eval_frequency})"
    )
    # ... training loop logic here


class TrainingBootstrap:
    """Bootstrap class for backward compatibility."""

    def start(self) -> None:
        """Start the training service."""
        bootstrap_training_service()


def main() -> None:
    """Main entry point for the training service bootstrap."""
    bootstrap_training_service()


if __name__ == "__main__":
    main()
