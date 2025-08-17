"""
Enhanced logging configuration schema for T005 standardization.

Provides Pydantic configuration models for the new ServiceLogger framework
while maintaining compatibility with existing logging infrastructure.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


# Renamed from EnhancedLoggingConfig to ServiceLoggingConfig for clarity
class ServiceLoggingConfig(BaseModel):
    """
    Enhanced configuration schema for standardized T005 logging.

    This configuration extends the existing LoggingConfig to support
    the new ServiceLogger framework with trading-specific features.
    """

    # Basic logging configuration (compatible with existing LoggingConfig)
    level: str = Field(
        default="INFO",
        description="Default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    format: str = Field(
        default="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        description="Log format string (used for human-readable logging)"
    )

    file_path: Optional[str] = Field(
        default=None,
        description="Custom log file path (optional)"
    )

    console_enabled: bool = Field(
        default=True,
        description="Enable console logging"
    )

    # T005 Enhanced features
    json_format: bool = Field(
        default=False,
        description="Force JSON formatting (auto-enabled in production/staging)"
    )

    # File logging configuration
    file_logging: bool = Field(
        default=True,
        description="Enable file logging with rotation"
    )

    max_file_size: int = Field(
        default=10485760,  # 10MB
        description="Maximum log file size in bytes before rotation"
    )

    backup_count: int = Field(
        default=5,
        description="Number of backup log files to keep"
    )

    rotation_enabled: bool = Field(
        default=True,
        description="Enable log file rotation"
    )

    max_bytes: int = Field(
        default=10485760,  # 10MB - alias for max_file_size
        description="Maximum log file size in bytes (legacy compatibility)"
    )

    # Third-party logger configurations
    third_party_loggers: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Configuration for third-party loggers (e.g., Flask, SQLAlchemy)"
    )

    # Performance settings
    sampling_rate: Optional[float] = Field(
        default=None,
        description="Log sampling rate for high-volume logs (0.0-1.0)"
    )

    # Abbreviation settings
    abbreviate_logger_names: bool = Field(
        default=True,
        description="Enable cached abbreviated logger name (short_name / short_logger)."
    )

    # Production-specific settings
    sentry_dsn: Optional[str] = Field(
        default=None,
        description="Sentry DSN for error tracking (future enhancement)"
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"

        # Configuration examples for documentation
        schema_extra = {
            "examples": [
                {
                    "level": "INFO",
                    "json_format": False,
                    "file_logging": True,
                    "max_file_size": 10485760,
                    "backup_count": 5,
                    "console_enabled": True,
                    "third_party_loggers": {
                        "flask": {"level": "WARNING"},
                        "sqlalchemy": {"level": "WARNING"},
                        "kafka": {"level": "INFO"}
                    }
                }
            ]
        }


# Renamed from configure_t005_logging to configure_service_logging for clarity
def configure_service_logging(
    service_name: str,
    stage: str,
    config: Optional[ServiceLoggingConfig] = None
) -> None:
    """
    Configure standardized logging for a DRL trading service.

    This function sets up the ServiceLogger framework with enhanced
    features including structured JSON logging and trading context tracking.

    Args:
        service_name: Name of the service (e.g., 'drl-trading-ingest')
        stage: Deployment stage ('local', 'cicd', 'prod')
        config: Optional ServiceLoggingConfig instance (uses defaults if not provided)

    Example:
        >>> from drl_trading_common.config.enhanced_logging_config import configure_service_logging
        >>> configure_service_logging('drl-trading-ingest', 'local')
    """
    from drl_trading_common.logging.service_logger import ServiceLogger

    service_logger = ServiceLogger(service_name, stage, config)
    service_logger.configure()
