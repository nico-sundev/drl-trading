"""Dependency injection module for execution service."""
from injector import Module, provider, singleton
from drl_trading_common.config.enhanced_service_config_loader import EnhancedServiceConfigLoader

# Use relative import to avoid module path issues
from ..config.execution_config import ExecutionConfig


class ExecutionModule(Module):
    """Dependency injection module for execution service."""

    def __init__(self, config: ExecutionConfig | None = None) -> None:
        """Initialize the module with optional configuration.

        Args:
            config: Optional ExecutionConfig. If not provided, will be loaded via EnhancedServiceConfigLoader
        """
        self._config = config

    @provider
    @singleton
    def provide_execution_config(self) -> ExecutionConfig:
        """Provide execution configuration."""
        if self._config is None:
            self._config = EnhancedServiceConfigLoader.load_config(ExecutionConfig)
        return self._config
