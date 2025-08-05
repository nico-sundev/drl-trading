"""
Standardized logging configuration for all DRL Trading services.

Provides consistent logging patterns across deployable services while
maintaining flexibility for different deployment environments.
"""
import logging
import logging.config
import os
from typing import Optional, Dict, Any


class StandardLoggingSetup:
    """Standardized logging configuration for all services."""

    DEFAULT_LOG_FORMAT = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "[%(filename)s:%(lineno)d] - %(message)s"
    )

    JSON_LOG_FORMAT = (
        '{"timestamp": "%(asctime)s", "service": "{service}", '
        '"level": "%(levelname)s", "logger": "%(name)s", '
        '"message": "%(message)s"}'
    )

    @classmethod
    def configure_logging(
        cls,
        service_name: str,
        config: Optional[Any] = None,
        log_level: Optional[str] = None
    ) -> None:
        """
        Configure standardized logging for a service.

        Args:
            service_name: Name of the service for log identification
            config: Optional logging configuration from service config
            log_level: Optional override for log level
        """
        # Determine log level from various sources
        log_level = cls._determine_log_level(config, log_level)

        # Get stage for environment-specific logging
        stage = os.environ.get("STAGE", "local")

        # Build logging configuration
        logging_config = cls._build_logging_config(service_name, log_level, stage)

        # Override with custom config if provided
        if config and hasattr(config, '__dict__'):
            # If config has logging attributes, use them
            if hasattr(config, 'level'):
                log_level = config.level
            # Update any file paths or other settings
            logging_config = cls._merge_custom_config(logging_config, config)

        # Apply configuration
        logging.config.dictConfig(logging_config)

        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        logger = logging.getLogger(service_name)
        logger.info(f"Logging configured for {service_name} service (level: {log_level}, stage: {stage})")

    @classmethod
    def _determine_log_level(
        cls,
        config: Optional[Any],
        log_level_override: Optional[str]
    ) -> str:
        """Determine log level from multiple sources with precedence."""
        # 1. Explicit override (highest priority)
        if log_level_override:
            return log_level_override.upper()

        # 2. Environment variable
        env_level = os.environ.get("LOG_LEVEL")
        if env_level:
            return env_level.upper()

        # 3. Service configuration
        if config and hasattr(config, 'level'):
            return config.level.upper()

        # 4. Default
        return "INFO"

    @classmethod
    def _build_logging_config(
        cls,
        service_name: str,
        log_level: str,
        stage: str
    ) -> Dict[str, Any]:
        """Build the logging configuration dictionary."""
        # Use JSON format for production, human-readable for local
        console_formatter = "json" if stage == "prod" else "standard"

        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": cls.DEFAULT_LOG_FORMAT,
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                },
                "json": {
                    "format": cls.JSON_LOG_FORMAT.format(service=service_name),
                    "datefmt": "%Y-%m-%dT%H:%M:%S"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": log_level,
                    "formatter": console_formatter,
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": log_level,
                    "formatter": "json",
                    "filename": f"logs/{service_name}.log",
                    "mode": "a",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5
                }
            },
            "loggers": {
                service_name: {
                    "level": log_level,
                    "handlers": ["console", "file"],
                    "propagate": False
                },
                "drl_trading_common": {
                    "level": log_level,
                    "handlers": ["console", "file"],
                    "propagate": False
                },
                "drl_trading_core": {
                    "level": log_level,
                    "handlers": ["console", "file"],
                    "propagate": False
                },
                # Add other DRL trading loggers
                "drl_trading_inference": {
                    "level": log_level,
                    "handlers": ["console", "file"],
                    "propagate": False
                },
                "drl_trading_training": {
                    "level": log_level,
                    "handlers": ["console", "file"],
                    "propagate": False
                },
                "drl_trading_ingest": {
                    "level": log_level,
                    "handlers": ["console", "file"],
                    "propagate": False
                },
                "drl_trading_execution": {
                    "level": log_level,
                    "handlers": ["console", "file"],
                    "propagate": False
                },
                "drl_trading_preprocess": {
                    "level": log_level,
                    "handlers": ["console", "file"],
                    "propagate": False
                }
            },
            "root": {
                "level": "WARNING",
                "handlers": ["console"]
            }
        }

    @classmethod
    def _merge_custom_config(
        cls,
        base_config: Dict[str, Any],
        custom_config: Any
    ) -> Dict[str, Any]:
        """Merge custom configuration with base configuration."""
        # For now, just return base config
        # In the future, this could be enhanced to merge specific attributes
        # from the custom config object
        return base_config
