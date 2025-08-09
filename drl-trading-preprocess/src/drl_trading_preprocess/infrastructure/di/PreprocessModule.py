"""Dependency injection module for preprocess service."""
from injector import Module, provider, singleton
from drl_trading_common.config.service_config_loader import ServiceConfigLoader

# Use relative import to avoid module path issues
from .infrastructure.config.preprocess_config import PreprocessConfig


class PreprocessModule(Module):
    """Dependency injection module for preprocess service."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        """Initialize the module with optional configuration.

        Args:
            config: Optional PreprocessConfig. If not provided, will be loaded via ServiceConfigLoader
        """
        self._config = config

    @provider
    @singleton
    def provide_preprocess_config(self) -> PreprocessConfig:
        """Provide preprocess configuration."""
        if self._config is None:
            self._config = ServiceConfigLoader.load_config(PreprocessConfig)
        return self._config
