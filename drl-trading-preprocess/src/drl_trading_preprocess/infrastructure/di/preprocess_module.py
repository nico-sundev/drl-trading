"""Dependency injection module for preprocess service (config injected)."""
from drl_trading_preprocess.infrastructure.config.preprocess_config import PreprocessConfig
from injector import Module, provider, singleton


class PreprocessModule(Module):
    """Dependency injection module for preprocess service.

    Expects the already-loaded config instance to be passed from bootstrap.
    """

    def __init__(self, config: PreprocessConfig) -> None:
        self._config = config

    @provider
    @singleton
    def provide_preprocess_config(self) -> PreprocessConfig:
        """Provide preprocess configuration (no reload)."""
        return self._config
