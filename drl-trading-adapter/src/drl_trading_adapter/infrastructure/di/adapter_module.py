from __future__ import annotations

from injector import Binder, Module, provider, singleton

from drl_trading_adapter.adapter.feature_store import (
    FeatureStoreFetchRepository,
)
from drl_trading_adapter.adapter.feature_store.offline import OfflineRepoStrategy
from drl_trading_adapter.adapter.feature_store.offline.offline_feature_repo_interface import (
    IOfflineFeatureRepository,
)
from drl_trading_adapter.adapter.feature_store.provider import (
    FeastProvider,
    FeatureStoreWrapper,
)
from drl_trading_adapter.adapter.feature_store.provider.mapper.feature_field_mapper import (
    FeatureFieldFactory,
    IFeatureFieldFactory,
)
from drl_trading_adapter.adapter.database import (
    MarketDataRepository,
    SQLAlchemySessionFactory,
)
from drl_trading_common.config.feature_config import FeatureStoreConfig
from drl_trading_core.core.port import IFeatureStoreFetchPort
from drl_trading_core.core.port.market_data_reader_port import MarketDataReaderPort


class AdapterModule(Module):
    def configure(self, binder: Binder) -> None:  # type: ignore[override]
        # Feature store bindings
        binder.bind(FeatureStoreWrapper, to=FeatureStoreWrapper, scope=singleton)
        binder.bind(FeastProvider, to=FeastProvider, scope=singleton)
        binder.bind(IFeatureFieldFactory, to=FeatureFieldFactory, scope=singleton)
        binder.bind(
            IFeatureStoreFetchPort, to=FeatureStoreFetchRepository, scope=singleton
        )

        # Market data database bindings
        binder.bind(SQLAlchemySessionFactory, to=SQLAlchemySessionFactory, scope=singleton)
        binder.bind(MarketDataReaderPort, to=MarketDataRepository, scope=singleton)

        # Repositories constructed via provider methods to avoid top-level concrete imports

    @provider  # type: ignore[misc]
    @singleton
    def provide_offline_feature_repository(
        self, feature_store_config: FeatureStoreConfig
    ) -> IOfflineFeatureRepository:
        strategy = OfflineRepoStrategy(feature_store_config)
        return strategy.create_offline_repository()
