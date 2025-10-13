"""Dependency injection module for preprocess service (config injected)."""

from drl_trading_common.config.feature_config import FeatureStoreConfig
from drl_trading_preprocess.adapter.feature_store.feature_store_save_repository import FeatureStoreSaveRepository
from drl_trading_preprocess.core.port.feature_store_save_port import IFeatureStoreSavePort
from drl_trading_preprocess.core.port.state_persistence_port import IStatePersistencePort
from drl_trading_preprocess.core.service.resample.state_persistence_service import StatePersistenceService
from drl_trading_preprocess.infrastructure.adapter.state_persistence.noop_state_persistence_service import NoOpStatePersistenceService
from drl_trading_preprocess.infrastructure.config.preprocess_config import PreprocessConfig
from injector import Binder, Module, provider, singleton


class PreprocessModule(Module):
    """Dependency injection module for preprocess service.

    Expects the already-loaded config instance to be passed from bootstrap.
    """

    def __init__(self, config: PreprocessConfig) -> None:
        self._config = config

    def configure(self, binder: Binder) -> None:  # type: ignore[override]
        binder.bind(
            IFeatureStoreSavePort,
            to=FeatureStoreSaveRepository,
            scope=singleton,
        )
        # Prevent auto-wiring of StatePersistenceService - use provider instead
        # This ensures Optional[StatePersistenceService] doesn't try to auto-instantiate

    @provider
    @singleton
    def provide_preprocess_config(self) -> PreprocessConfig:
        """Provide preprocess configuration (no reload)."""
        return self._config

    @provider
    @singleton
    def provide_feature_store_config(self) -> FeatureStoreConfig:
        """Provide feature store configuration (no reload)."""
        return self._config.feature_store_config

    @provider  # type: ignore[misc]
    @singleton
    def provide_state_persistence_service(self) -> IStatePersistencePort:
        """
        Provide state persistence service based on configuration.

        Returns:
            StatePersistenceService if enabled, NoOpStatePersistenceService if disabled

        Note: Always returns a valid implementation (Null Object Pattern).
        When disabled, returns no-op implementation that safely does nothing.
        """
        if not self._config.resample_config.state_persistence_enabled:
            return NoOpStatePersistenceService()

        return StatePersistenceService(
            state_file_path=self._config.resample_config.state_file_path,
            backup_interval=self._config.resample_config.state_backup_interval
        )
