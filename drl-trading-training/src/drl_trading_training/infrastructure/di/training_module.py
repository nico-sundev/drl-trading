"""Dependency injection module for training service."""
from injector import Module, provider, singleton
from drl_trading_common.config.service_config_loader import ServiceConfigLoader

# Use relative import to avoid module path issues
from ..config.training_config import TrainingConfig


class TrainingModule(Module):
    """Dependency injection module for training service."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        """Initialize the module with optional configuration.

        Args:
            config: Optional TrainingConfig. If not provided, will be loaded via ServiceConfigLoader
        """
        self._config = config

    @provider
    @singleton
    def provide_training_config(self) -> TrainingConfig:
        """Provide training configuration."""
        if self._config is None:
            self._config = ServiceConfigLoader.load_config(TrainingConfig)
        return self._config
