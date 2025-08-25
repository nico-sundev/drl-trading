from __future__ import annotations

from injector import Binder, Module, provider, singleton

from drl_trading_adapter.adapter.feature_store import (
    FeastProvider,
    FeatureStoreFetchAdapter,
    FeatureStoreWrapper,
)
from drl_trading_adapter.adapter.feature_store.offline import OfflineRepoStrategy
from drl_trading_adapter.adapter.feature_store.offline.offline_feature_repo_interface import (
    IOfflineFeatureRepository,
)
from drl_trading_common.config.feature_config import FeatureStoreConfig
from drl_trading_core.common.core.port import IFeatureStoreFetchPort


class AdapterModule(Module):
    def configure(self, binder: Binder) -> None:  # type: ignore[override]
        binder.bind(FeatureStoreWrapper, to=FeatureStoreWrapper, scope=singleton)
        binder.bind(FeastProvider, to=FeastProvider, scope=singleton)
        binder.bind(
            IFeatureStoreFetchPort, to=FeatureStoreFetchAdapter, scope=singleton
        )
        # Repositories constructed via provider methods to avoid top-level concrete imports

    @provider  # type: ignore[misc]
    @singleton
    def provide_offline_feature_repository(
        self, feature_store_config: FeatureStoreConfig
    ) -> IOfflineFeatureRepository:
        strategy = OfflineRepoStrategy(feature_store_config)
        return strategy.create_offline_repository()
