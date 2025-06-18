"""Logging configuration utilities for the DRL Trading system.

This module provides functions to configure logging for both simple default setup
and advanced configuration-driven setups across different services.

Three main configuration functions are available:
- configure_unified_logging: Main entry point that adapts to available config
- configure_service_logging: Advanced logging with infrastructure config
- configure_logging: Simple logging with reasonable defaults

Example usage:
    # Basic usage with defaults
    from drl_trading_common.config.logging_config import configure_unified_logging
    configure_unified_logging(service_name="my_service")

    # With infrastructure config
    from drl_trading_common.config.logging_config import configure_unified_logging
    from my_service.config import ServiceConfig

    config = ServiceConfig.load()
    configure_unified_logging(config, service_name="my_service")
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, List

from drl_trading_common.config.infrastructure_config import LoggingConfig


def configure_service_logging(logging_config: LoggingConfig, service_name: str) -> None:
    """Configure logging based on service-specific infrastructure config.

    Args:
        logging_config: The logging configuration from infrastructure config
        service_name: The name of the service for log file naming
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Determine log level from config
    level = getattr(logging, logging_config.level.upper(), logging.INFO)

    # Determine log file path
    log_file = log_dir / f"{service_name}.log"
    if logging_config.file_path:
        log_file = Path(logging_config.file_path)
        # Ensure parent directory exists
        log_file.parent.mkdir(exist_ok=True, parents=True)

    # Reset root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure handlers
    handlers: List[logging.Handler] = []
      # Add file handler with optional rotation
    try:
        if getattr(logging_config, "rotation_enabled", True):
            # Get rotation parameters from config or use defaults
            max_bytes = getattr(logging_config, "max_bytes", 10_485_760)
            backup_count = getattr(logging_config, "backup_count", 5)

            handlers.append(
                RotatingFileHandler(
                    str(log_file),
                    maxBytes=max_bytes,
                    backupCount=backup_count
                )
            )
        else:
            # Use standard file handler without rotation if explicitly disabled
            handlers.append(logging.FileHandler(str(log_file)))
    except Exception:
        # Fall back to standard file handler if rotation isn't available
        handlers.append(logging.FileHandler(str(log_file)))

    # Add console handler if enabled
    if logging_config.console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        # Set formatter for console handler
        formatter = logging.Formatter(logging_config.format)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    # Configure root logger with settings from config
    logging.basicConfig(
        level=level,
        format=logging_config.format,
        handlers=handlers,
    )


def configure_unified_logging(config: Any = None, service_name: str = "drl_trading_common") -> None:
    """Unified logging configuration function that adapts to available config.

    This function is the recommended entry point for logging configuration.
    It can work with any configuration object that has the right structure,
    or fall back to reasonable defaults.

    Args:
        config: Any configuration object with infrastructure.logging attributes
        service_name: Name of the service for log file naming

    Returns:
        None

    Note:
        This function will never raise an exception. If configuration fails,
        it will fall back to a basic working logging configuration.
    """
    # First configure a simple stderr logger for bootstrap errors
    bootstrap_logger = logging.getLogger("bootstrap")
    if not bootstrap_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        bootstrap_logger.addHandler(handler)

    try:
        # Try to extract logging config from the provided config object
        if config:
            if not hasattr(config, "infrastructure"):
                bootstrap_logger.warning("Config object has no 'infrastructure' attribute, using default logging")
                configure_logging(service_name)
                return

            if not hasattr(config.infrastructure, "logging"):
                bootstrap_logger.warning("Config object has no 'infrastructure.logging' attribute, using default logging")
                configure_logging(service_name)
                return

            # Try to get service name from config
            if hasattr(config.infrastructure, "service_name"):
                service_name = config.infrastructure.service_name

            # Configure with infrastructure config
            bootstrap_logger.debug(f"Configuring logging for service {service_name} using infrastructure config")
            configure_service_logging(config.infrastructure.logging, service_name)
            return
        else:
            bootstrap_logger.info(f"No config provided, using default logging for service {service_name}")
    except Exception as e:
        # If anything fails, log the error and fall back to simple config
        bootstrap_logger.error(f"Error configuring logging from config: {e}. Using default configuration.")

    # Fall back to simple configuration
    configure_logging(service_name)


def configure_logging(service_name: str = "drl_trading_common") -> None:
    """Configure logging for the application with default settings.

    This is a simpler version that doesn't require a LoggingConfig object.

    Args:
        service_name: Optional name of the service for log file naming
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"{service_name}.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
