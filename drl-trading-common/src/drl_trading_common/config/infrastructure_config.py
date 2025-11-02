"""Infrastructure configuration for deployment-specific settings."""
from typing import Optional
from drl_trading_common.base.base_schema import BaseSchema
from drl_trading_common.config.kafka_config import KafkaConnectionConfig
from drl_trading_common.config.resilience_config import ResilienceConfig

class DatabaseConfig(BaseSchema):
    """Database configuration."""
    provider: str = "postgresql"  # postgresql | sqlite
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None


class LoggingConfig(BaseSchema):
    """Logging configuration.

    Attributes:
        level: Logging level (INFO, DEBUG, WARNING, ERROR, CRITICAL)
        format: Log message format string
        file_path: Path to log file (will be created if it doesn't exist)
        console_enabled: Whether to also log to console
        rotation_enabled: Whether to use rotating log files
        max_bytes: Maximum size of log file before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
    """
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    file_path: Optional[str] = "logs/application.log"
    console_enabled: bool = True
    rotation_enabled: bool = True
    max_bytes: int = 10_485_760  # 10 MB
    backup_count: int = 5


class MonitoringConfig(BaseSchema):
    """Monitoring and observability configuration."""
    prometheus_enabled: bool = False
    prometheus_port: Optional[int] = 9090
    jaeger_enabled: bool = False
    jaeger_endpoint: Optional[str] = None


class WebApiConfig(BaseSchema):
    """Web API configuration for Flask/FastAPI services."""
    port: int = 8080
    host: str = "0.0.0.0"
    debug: bool = False
    threaded: bool = True
    max_content_length: int = 16 * 1024 * 1024  # 16MB


class InfrastructureConfig(BaseSchema):
    """Infrastructure configuration for deployment environment."""
    database: DatabaseConfig = DatabaseConfig()
    logging: LoggingConfig = LoggingConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    webapi: Optional[WebApiConfig] = None
    kafka: Optional[KafkaConnectionConfig] = None
    resilience: Optional[ResilienceConfig] = None
