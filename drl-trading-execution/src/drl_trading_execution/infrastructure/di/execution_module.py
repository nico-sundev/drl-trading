"""Dependency injection module for execution service (config injected)."""
from injector import Module, provider, singleton

# Use relative import to avoid module path issues
from ..config.execution_config import ExecutionConfig


class ExecutionModule(Module):
    """Dependency injection module for execution service.

    Expects the already-loaded config instance to be passed from bootstrap.
    """

    def __init__(self, config: ExecutionConfig) -> None:
        self._config = config

    @provider
    @singleton
    def provide_execution_config(self) -> ExecutionConfig:
        """Provide execution configuration (no reload)."""
        return self._config
