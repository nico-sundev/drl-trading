"""Dependency injection module for training service (config injected)."""
from injector import Module, provider, singleton

# Use relative import to avoid module path issues
from ..config.training_config import TrainingConfig


class TrainingModule(Module):
    """Dependency injection module for training service.

    Expects the already-loaded config instance from bootstrap.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self._config = config

    @provider
    @singleton
    def provide_training_config(self) -> TrainingConfig:
        """Provide training configuration (no reload)."""
        return self._config
