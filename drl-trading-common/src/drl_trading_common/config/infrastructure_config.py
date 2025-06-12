"""Infrastructure configuration for deployment-specific settings."""
from typing import Optional
from drl_trading_common.base.base_schema import BaseSchema


class MessagingConfig(BaseSchema):
    """Message bus configuration."""
    provider: str = "rabbitmq"  # rabbitmq | in_memory
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    vhost: Optional[str] = None


class DatabaseConfig(BaseSchema):
    """Database configuration."""
    provider: str = "postgresql"  # postgresql | sqlite
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None


class LoggingConfig(BaseSchema):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    file_path: Optional[str] = "logs/application.log"
    console_enabled: bool = True


class MonitoringConfig(BaseSchema):
    """Monitoring and observability configuration."""
    prometheus_enabled: bool = False
    prometheus_port: Optional[int] = 9090
    jaeger_enabled: bool = False
    jaeger_endpoint: Optional[str] = None


class InfrastructureConfig(BaseSchema):
    """Infrastructure configuration for deployment environment."""
    deployment_mode: str = "development"  # development | staging | production
    messaging: MessagingConfig = MessagingConfig()
    database: DatabaseConfig = DatabaseConfig()
    logging: LoggingConfig = LoggingConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
