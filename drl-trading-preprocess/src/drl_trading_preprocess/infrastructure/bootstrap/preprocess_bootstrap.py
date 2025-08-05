"""Bootstrap implementation for preprocess service following T004 patterns."""
import logging
from typing import Optional

from drl_trading_common.config.logging_config import configure_unified_logging
from drl_trading_common.config.enhanced_service_config_loader import EnhancedServiceConfigLoader

# Use relative import to avoid module path issues
from ..config.preprocess_config import PreprocessConfig

# Configure basic logging for bootstrap phase
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def setup_logging(config: Optional[PreprocessConfig] = None) -> None:
    """Set up logging using the unified configuration approach.

    Args:
        config: Optional PreprocessConfig instance with logging settings
    """
    configure_unified_logging(config, service_name="preprocess")
    logger.info("Logging configured using unified configuration approach")


def bootstrap_preprocess_service() -> None:
    """Bootstrap the preprocess service with proper configuration."""
    try:
        # Load configuration using lean EnhancedServiceConfigLoader
        logger.info("Loading configuration with lean EnhancedServiceConfigLoader")
        config = EnhancedServiceConfigLoader.load_config(PreprocessConfig)

        # Reconfigure logging with loaded config
        setup_logging(config)

        # Log effective configuration for debugging
        logger.info(
            f"Preprocess service initialized in {config.stage} mode "
            f"for {config.app_name} v{config.version}"
        )

        # Configure service components based on the config
        setup_data_sources(config)
        setup_feature_engineering(config)
        setup_messaging(config)
        setup_monitoring(config)

        # Start processing
        start_processing_loop(config)

    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        exit(1)
    except Exception as e:
        logger.exception(f"Failed to initialize preprocess service: {e}")
        exit(2)


def setup_data_sources(config: PreprocessConfig) -> None:
    """Set up data source connections and validation."""
    input_path = config.data_source.input_path
    formats = config.data_source.supported_formats
    batch_size = config.data_source.batch_size

    logger.info(
        f"Setting up data sources from {input_path} "
        f"(formats: {formats}, batch_size: {batch_size})"
    )
    # ... data source setup logic here


def setup_feature_engineering(config: PreprocessConfig) -> None:
    """Set up feature engineering pipeline."""
    enabled_features = config.feature_engineering.enabled_features
    lookback = config.feature_engineering.lookback_period
    scaling = config.feature_engineering.scaling_method

    logger.info(
        f"Setting up feature engineering: {enabled_features} "
        f"(lookback: {lookback}, scaling: {scaling})"
    )
    # ... feature engineering setup logic here


def setup_messaging(config: PreprocessConfig) -> None:
    """Set up message bus connections."""
    provider = config.infrastructure.messaging.provider

    logger.info(f"Setting up messaging with {provider} provider")
    # ... messaging setup logic here


def setup_monitoring(config: PreprocessConfig) -> None:
    """Set up monitoring and health checks."""
    logger.info("Setting up monitoring and health checks")
    # ... monitoring setup logic here


def start_processing_loop(config: PreprocessConfig) -> None:
    """Start the main data processing loop."""
    output_path = config.output.output_path
    output_format = config.output.format
    validation = config.output.validation_enabled

    logger.info(
        f"Starting data processing pipeline "
        f"(output: {output_path}, format: {output_format}, validation: {validation})"
    )
    # ... processing loop logic here


class PreprocessBootstrap:
    """Bootstrap class for backward compatibility."""

    def start(self) -> None:
        """Start the preprocess service."""
        bootstrap_preprocess_service()


def main() -> None:
    """Main entry point for the preprocess service bootstrap."""
    bootstrap_preprocess_service()


if __name__ == "__main__":
    main()
