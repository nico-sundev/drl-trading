"""Dependency injection module for execution service."""
from injector import Module, provider, singleton
from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.config.infrastructure_config import MessagingConfig, DatabaseConfig, LoggingConfig


class ExecutionModule(Module):
    """Dependency injection module for execution service."""

    @provider
    @singleton
    def provide_application_config(self) -> BaseApplicationConfig:
        """Provide application configuration."""
        # This will be implemented with proper config loading
        return BaseApplicationConfig(app_name="drl-trading-execution")

    @provider
    @singleton
    def provide_messaging_config(self) -> MessagingConfig:
        """Provide messaging configuration."""
        return MessagingConfig()

    @provider
    @singleton
    def provide_database_config(self) -> DatabaseConfig:
        """Provide database configuration."""
        return DatabaseConfig()

    @provider
    @singleton
    def provide_logging_config(self) -> LoggingConfig:
        """Provide logging configuration."""
        return LoggingConfig()
