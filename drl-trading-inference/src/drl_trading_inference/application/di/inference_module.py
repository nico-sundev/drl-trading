"""Dependency injection module for inference service."""
from injector import Module, provider, singleton
from drl_trading_inference.application.config.inference_config import InferenceConfig
from drl_trading_common.config.infrastructure_config import LoggingConfig


class InferenceModule(Module):
    """Dependency injection module for inference service."""

    def __init__(self, config: InferenceConfig):
        """Initialize module with configuration."""
        self.config = config

    @provider
    @singleton
    def provide_inference_config(self) -> InferenceConfig:
        """Provide inference service configuration."""
        return self.config

    @provider
    @singleton
    def provide_logging_config(self) -> LoggingConfig:
        """Provide logging configuration."""
        return self.config.infrastructure.logging
