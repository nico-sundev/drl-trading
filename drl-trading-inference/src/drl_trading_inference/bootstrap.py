"""Example of using the EnhancedServiceConfigLoader in a microservice."""
import logging
import os
from typing import Optional

from drl_trading_common.config.logging_config import configure_unified_logging
from drl_trading_common.config.enhanced_service_config_loader import EnhancedServiceConfigLoader

# Local import with proper type ignoring for mypy during development
from drl_trading_inference.config.inference_config import (
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
        # Prioritize environment variable if present
        config_path = os.environ.get("SERVICE_CONFIG_PATH")
        if config_path:
            logger.info(f"Loading configuration from SERVICE_CONFIG_PATH: {config_path}")
            config = EnhancedServiceConfigLoader.load_config(
                InferenceConfig,
                config_path=config_path,
                secret_substitution=True,
                env_override=True
            )
        else:
            # Use service-specific logic to locate config
            logger.info("Discovering configuration for inference service")
            config = EnhancedServiceConfigLoader.load_config(
                InferenceConfig,
                service="inference",
                secret_substitution=True,
                env_override=True
            )

        # Now that we have the config, reconfigure logging properly
        setup_logging(config)

        # Get service name for logging
        service_name = getattr(config.infrastructure, "service_name", "inference")

        # Log effective configuration for debugging
        logger.info(
            f"Inference service initialized in {config.stage} mode "
            f"for {service_name} v{getattr(config.infrastructure, 'version', '1.0.0')}"
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
    # Example of accessing typed configuration
    model_path = config.model_config["model_path"]
    model_format = config.model_config["model_format"]

    logger.info(f"Loading model from {model_path} in {model_format} format")
    # ... model loading logic here


def setup_messaging(config: InferenceConfig) -> None:
    """Set up message bus connections."""
    # Example of combining infrastructure and service-specific config
    # provider = config.infrastructure.message_bus.provider
    input_topic = config.message_routing["input_topic"]
    output_topic = config.message_routing["output_topic"]

    logger.info(
        # f"Setting up messaging with {provider} provider. "
        f"Subscribing to {input_topic}, publishing to {output_topic}"
    )
    # ... messaging setup logic here


def setup_monitoring(config: InferenceConfig) -> None:
    """Set up monitoring and health checks."""
    if config.infrastructure.monitoring.prometheus_enabled:
        port = config.infrastructure.monitoring.prometheus_port
        logger.info(f"Enabling Prometheus metrics endpoint on port {port}")

        try:
            # This would typically import and configure prometheus client
            # Example implementation (depends on actual prometheus library)
            logger.info("Initializing prometheus metrics endpoint")

            # Register custom metrics (examples)
            logger.info("Registering inference service metrics")

            # Start metrics server if needed
            logger.info(f"Starting metrics server on port {port}")
        except ImportError:
            logger.warning("Prometheus client library not installed. Metrics disabled.")
        except Exception as e:
            logger.error(f"Failed to initialize monitoring: {e}")
            # Don't fail the application if monitoring setup fails


def start_inference_loop(config: InferenceConfig) -> None:
    """Start the main inference processing loop."""
    prediction_frequency = config.processing_config["prediction_frequency"]
    logger.info(f"Starting inference loop with {prediction_frequency} prediction frequency")
    # ... inference loop logic here

def main() -> None:
    """Main entry point for the inference service bootstrap."""
    bootstrap_inference_service()

if __name__ == "__main__":
    main()
