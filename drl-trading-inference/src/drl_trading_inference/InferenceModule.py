"""Dependency injection module for inference service."""
from injector import Module, provider, singleton
from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.config.infrastructure_config import MessagingConfig, LoggingConfig


class InferenceModule(Module):
    """Dependency injection module for inference service."""

    @provider
    @singleton
    def provide_application_config(self) -> BaseApplicationConfig:
        """Provide application configuration."""
        return BaseApplicationConfig(app_name="drl-trading-inference")

    @provider
    @singleton
    def provide_messaging_config(self) -> MessagingConfig:
        """Provide messaging configuration."""
        return MessagingConfig()

    @provider
    @singleton
    def provide_logging_config(self) -> LoggingConfig:
        """Provide logging configuration."""
        return LoggingConfig()
