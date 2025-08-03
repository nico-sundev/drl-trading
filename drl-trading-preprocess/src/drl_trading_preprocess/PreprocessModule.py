"""Dependency injection module for preprocess service."""
from injector import Module, provider, singleton
from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.config.infrastructure_config import MessagingConfig, DatabaseConfig, LoggingConfig


class PreprocessModule(Module):
    """Dependency injection module for preprocess service."""

    @provider
    @singleton
    def provide_application_config(self) -> BaseApplicationConfig:
        """Provide application configuration."""
        return BaseApplicationConfig(app_name="drl-trading-preprocess")

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
