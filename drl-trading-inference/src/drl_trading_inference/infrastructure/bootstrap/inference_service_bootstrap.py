"""Example of using the EnhancedServiceConfigLoader in a microservice."""
import logging
from typing import Optional

from drl_trading_common.config.logging_config import configure_unified_logging
from drl_trading_common.config.enhanced_service_config_loader import EnhancedServiceConfigLoader

# Local import with proper type ignoring for mypy during development
from drl_trading_inference.infrastructure.config.inference_config import (
    InferenceConfig,  # type: ignore
)

# Configure basic logging for bootstrap phase
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def setup_logging(config: Optional[InferenceConfig] = None) -> None:
    """Set up logging using the appropriate method based on available config.

    If config is provided, uses the infrastructure logging configuration.
    Otherwise, falls back to the standard configure_logging from common library.

    Args:
        config: Optional InferenceConfig instance with logging settings
    """
    # Use the unified logging configuration function from the common library
    configure_unified_logging(config, service_name="inference")
    logger.info("Logging configured using unified configuration approach")


def bootstrap_inference_service() -> None:
    """Bootstrap the inference service with proper configuration."""
    # Load configuration with smart path discovery and environment detection
    try:
        # Use lean EnhancedServiceConfigLoader
        # Loads: application.yaml + application-{STAGE}.yaml + secret substitution
        logger.info("Loading configuration with lean EnhancedServiceConfigLoader")
        config = EnhancedServiceConfigLoader.load_config(InferenceConfig)

        # Now that we have the config, reconfigure logging properly
        setup_logging(config)

        # Log effective configuration for debugging
        logger.info(
            f"Inference service initialized in {config.stage} mode "
            f"for {config.app_name} v{config.version}"
        )

        # Configure service components based on the config
        setup_model(config)
        setup_messaging(config)
        setup_monitoring(config)

        # Start processing
        start_inference_loop(config)

    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        # Exit with error code
        exit(1)
    except Exception as e:
        logger.exception(f"Failed to initialize inference service: {e}")
        exit(2)


def setup_model(config: InferenceConfig) -> None:
    """Set up the ML model for inference."""
    # Use the correct config structure
    model_path = config.model.model_path
    model_name = config.model.model_name
    device = config.model.device

    logger.info(f"Loading model '{model_name}' from {model_path} on device: {device}")
    # ... model loading logic here


def setup_messaging(config: InferenceConfig) -> None:
    """Set up message bus connections."""
    # Use the messaging config from infrastructure
    provider = config.infrastructure.messaging.provider

    logger.info(f"Setting up messaging with {provider} provider")
    # ... messaging setup logic here


def setup_monitoring(config: InferenceConfig) -> None:
    """Set up monitoring and health checks."""
    # Note: monitoring config may not be implemented yet in infrastructure
    logger.info("Setting up monitoring and health checks")
    # ... monitoring setup logic here


def start_inference_loop(config: InferenceConfig) -> None:
    """Start the main inference processing loop."""
    endpoint = config.prediction.endpoint
    max_concurrent = config.prediction.max_concurrent_requests
    timeout = config.prediction.timeout_seconds

    logger.info(
        f"Starting inference service on endpoint {endpoint} "
        f"(max concurrent: {max_concurrent}, timeout: {timeout}s)"
    )
    # ... inference loop logic here

def main() -> None:
    """Main entry point for the inference service bootstrap."""
    bootstrap_inference_service()

if __name__ == "__main__":
    main()
