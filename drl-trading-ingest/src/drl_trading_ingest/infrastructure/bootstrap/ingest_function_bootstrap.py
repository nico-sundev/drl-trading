"""Function-based bootstrap for ingest service following T004 patterns."""
import logging
from typing import Optional

from drl_trading_common.config.logging_config import configure_unified_logging
from drl_trading_common.config.enhanced_service_config_loader import EnhancedServiceConfigLoader

# Import configuration
from drl_trading_ingest.infrastructure.config.ingest_config import IngestConfig

# Configure basic logging for bootstrap phase
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def setup_logging(config: Optional[IngestConfig] = None) -> None:
    """Set up logging configuration."""
    try:
        configure_unified_logging(
            format_string=config.infrastructure.logging.format if config else None,
            file_path=config.infrastructure.logging.file_path if config else None,
            console_enabled=config.infrastructure.logging.console_enabled if config else True,
        )
        logger.info("Logging configured using unified configuration approach")
    except Exception as e:
        logger.warning(f"Failed to configure logging: {e}, using default configuration")


def bootstrap_ingest_service() -> None:
    """
    Bootstrap the ingest service with T004 compliance.

    Follows the standardized bootstrap pattern with:
    - Lean configuration loading
    - Unified logging setup
    - Service-specific initialization
    """
    logger.info("Starting ingest service bootstrap")

    try:
        # Use lean EnhancedServiceConfigLoader
        # Loads: application.yaml + application-{STAGE}.yaml + secret substitution + .env
        logger.info("Loading configuration with lean EnhancedServiceConfigLoader")
        config = EnhancedServiceConfigLoader.load_config(IngestConfig)

        # Now that we have the config, reconfigure logging properly
        setup_logging(config)

        # Log effective configuration for debugging
        logger.info(
            f"Ingest service initialized in {config.stage} mode "
            f"for {config.app_name} v{config.version}"
        )

        # Set up data sources
        data_source = config.data_source
        logger.info(
            f"Setting up data sources: MT5={'enabled' if data_source.mt5_enabled else 'disabled'}, "
            f"Binance={'enabled' if data_source.binance_enabled else 'disabled'}"
        )
        if data_source.mt5_enabled:
            logger.info(f"MT5 symbols: {data_source.mt5_symbols} (timeframes: {data_source.mt5_timeframes})")

        # Set up message routing
        routing = config.message_routing
        logger.info(
            f"Setting up message routing: market_data={routing.market_data_topic}, "
            f"heartbeat={routing.heartbeat_topic}, errors={routing.error_topic}"
        )

        # Set up messaging infrastructure
        messaging = config.infrastructure.messaging
        logger.info(f"Setting up messaging with {messaging.provider} provider")

        # Set up data validation
        validation = config.data_validation
        logger.info(
            f"Setting up data validation: {'enabled' if validation.enable_validation else 'disabled'} "
            f"(outlier detection: {'enabled' if validation.enable_outlier_detection else 'disabled'})"
        )

        # Set up monitoring and health checks
        logger.info("Setting up monitoring and health checks")

        # Start the ingestion pipeline
        logger.info("Starting data ingestion pipeline")
        # Here we would start the actual data ingestion logic
        # For now, just log that we're ready
        logger.info("Ingest service is ready and running")

        # Keep the service alive
        logger.info("Service started successfully - press Ctrl+C to stop")
        import time
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")

    except Exception as e:
        logger.error(f"Failed to bootstrap ingest service: {e}")
        raise


def main() -> None:
    """Main entry point for the ingest service bootstrap."""
    bootstrap_ingest_service()


if __name__ == "__main__":
    main()
