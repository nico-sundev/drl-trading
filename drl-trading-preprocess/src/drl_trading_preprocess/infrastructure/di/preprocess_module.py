"""Dependency injection module for preprocess service (config injected)."""
from typing import Optional

from drl_trading_common.config.feature_config import FeatureStoreConfig
from drl_trading_preprocess.adapter.feature_store.feature_store_save_repository import FeatureStoreSaveRepository
from drl_trading_preprocess.core.port.feature_store_save_port import IFeatureStoreSavePort
from drl_trading_preprocess.core.service.resample.state_persistence_service import StatePersistenceService
from drl_trading_preprocess.infrastructure.config.preprocess_config import PreprocessConfig
from injector import Binder, Module, provider, singleton


class PreprocessModule(Module):
    """Dependency injection module for preprocess service.

    Expects the already-loaded config instance to be passed from bootstrap.
    """

    def __init__(self, config: PreprocessConfig) -> None:
        self._config = config

    def configure(self, binder: Binder) -> None:
        binder.bind(
            IFeatureStoreSavePort,
            to=FeatureStoreSaveRepository,
            scope=singleton,
        )

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

    @provider
    @singleton
    def provide_state_persistence_service(self) -> Optional[StatePersistenceService]:
        """
        Conditionally provide StatePersistenceService based on configuration.

        Returns:
            StatePersistenceService instance if state persistence is enabled, None otherwise
        """
        if self._config.resample_config.state_persistence_enabled:
            return StatePersistenceService(
                state_file_path=self._config.resample_config.state_file_path,
                backup_interval=self._config.resample_config.state_backup_interval
            )
        return None
