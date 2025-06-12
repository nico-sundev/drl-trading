"""Example of using the ServiceConfigLoader in a microservice."""
import logging
import os

from drl_trading_common.config.service_config_loader import ServiceConfigLoader
from drl_trading_inference.config.inference_config import InferenceConfig

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def bootstrap_inference_service():
    """Bootstrap the inference service with proper configuration."""
    # Load configuration with smart path discovery and environment detection
    try:
        # Prioritize environment variable if present
        config_path = os.environ.get("SERVICE_CONFIG_PATH")
        if config_path:
            logger.info(f"Loading configuration from SERVICE_CONFIG_PATH: {config_path}")
            config = ServiceConfigLoader.load_config(
                InferenceConfig,
                config_path=config_path
            )
        else:
            # Use service-specific logic to locate config
            logger.info("Discovering configuration for inference service")
            config = ServiceConfigLoader.load_config(
                InferenceConfig,
                service="inference"
            )

        # Log effective configuration for debugging
        logger.info(
            f"Inference service initialized in {config.infrastructure.deployment_mode} mode "
            f"for {config.infrastructure.service_name} v{config.infrastructure.version}"
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


def setup_model(config: InferenceConfig):
    """Set up the ML model for inference."""
    # Example of accessing typed configuration
    model_path = config.model_config["model_path"]
    model_format = config.model_config["model_format"]

    logger.info(f"Loading model from {model_path} in {model_format} format")
    # ... model loading logic here


def setup_messaging(config: InferenceConfig):
    """Set up message bus connections."""
    # Example of combining infrastructure and service-specific config
    provider = config.infrastructure.message_bus.provider
    input_topic = config.message_routing["input_topic"]
    output_topic = config.message_routing["output_topic"]

    logger.info(
        f"Setting up messaging with {provider} provider. "
        f"Subscribing to {input_topic}, publishing to {output_topic}"
    )
    # ... messaging setup logic here


def setup_monitoring(config: InferenceConfig):
    """Set up monitoring and health checks."""
    if config.infrastructure.monitoring.prometheus_enabled:
        port = config.infrastructure.monitoring.prometheus_port
        logger.info(f"Enabling Prometheus metrics endpoint on port {port}")
        # ... prometheus setup logic here


def start_inference_loop(config: InferenceConfig):
    """Start the main inference processing loop."""
    prediction_frequency = config.processing_config["prediction_frequency"]
    logger.info(f"Starting inference loop with {prediction_frequency} prediction frequency")
    # ... inference loop logic here


if __name__ == "__main__":
    bootstrap_inference_service()
