"""Adapter DI module binding adapter implementations to core ports.

Install AFTER CoreModule.
"""
from __future__ import annotations

from injector import Module, Binder, singleton, provider

from drl_trading_common.config.feature_config import FeatureStoreConfig
from drl_trading_core.preprocess.feature_store.port.feature_store_provider_port import (
    IFeatureStoreProvider,
)
from drl_trading_core.preprocess.feature_store.port.feature_store_operation_ports import (
    IFeatureViewFactory,
    IFeatureDefinitionApplier,
    IFeatureMaterializer,
    IOnlineFeatureWriter,
    IOnlineFeatureReader,
    IHistoricalFeatureReader,
)
from drl_trading_core.preprocess.feature_store.port.offline_feature_repo_interface import (
    IOfflineFeatureRepository,
)
from drl_trading_core.preprocess.feature_store.repository.feature_store_fetch_repo import (
    IFeatureStoreFetchRepository,
)
from drl_trading_core.preprocess.feature_store.repository.feature_store_save_repo import (
    IFeatureStoreSaveRepository,
)

from drl_trading_adapter.adapter.feature_store.feast.feast_provider import FeastProvider
from drl_trading_adapter.adapter.feature_store.feast.feature_store_wrapper import FeatureStoreWrapper
from drl_trading_adapter.adapter.feature_store.offline.offline_repo_strategy import OfflineRepoStrategy
from drl_trading_adapter.adapter.feature_store.feast.ports.feast_port_implementations import (
    FeastFeatureViewFactory,
    FeastFeatureDefinitionApplier,
    FeastFeatureMaterializer,
    FeastOnlineFeatureWriter,
    FeastOnlineFeatureReader,
    FeastHistoricalFeatureReader,
)
from drl_trading_core.preprocess.feature_store.mapper.feature_view_name_mapper import (
    FeatureViewNameMapper,
)


class AdapterModule(Module):
    def configure(self, binder: Binder) -> None:  # type: ignore[override]
        binder.bind(FeatureStoreWrapper, to=FeatureStoreWrapper, scope=singleton)
        binder.bind(IFeatureStoreProvider, to=FeastProvider, scope=singleton)
        # Granular port bindings
        binder.bind(IFeatureViewFactory, to=FeastFeatureViewFactory, scope=singleton)
        binder.bind(IFeatureDefinitionApplier, to=FeastFeatureDefinitionApplier, scope=singleton)
        binder.bind(IFeatureMaterializer, to=FeastFeatureMaterializer, scope=singleton)
        binder.bind(IOnlineFeatureWriter, to=FeastOnlineFeatureWriter, scope=singleton)
        binder.bind(IOnlineFeatureReader, to=FeastOnlineFeatureReader, scope=singleton)
        binder.bind(IHistoricalFeatureReader, to=FeastHistoricalFeatureReader, scope=singleton)
        # Repositories constructed via provider methods to avoid top-level concrete imports

    @provider  # type: ignore[misc]
    @singleton
    def provide_offline_feature_repository(
        self, feature_store_config: FeatureStoreConfig
    ) -> IOfflineFeatureRepository:
        strategy = OfflineRepoStrategy(feature_store_config)
        return strategy.create_offline_repository()

    @provider  # type: ignore[misc]
    @singleton
    def provide_feature_store_save_repository(
        self,
        feature_view_factory: IFeatureViewFactory,
        definition_applier: IFeatureDefinitionApplier,
        materializer: IFeatureMaterializer,
        online_feature_writer: IOnlineFeatureWriter,
        offline_feature_repository: IOfflineFeatureRepository,
        feature_store_config: FeatureStoreConfig,
        feature_view_name_mapper: FeatureViewNameMapper,
    ) -> IFeatureStoreSaveRepository:  # noqa: D401
        # Local import to keep concrete class out of module interface
        from drl_trading_core.preprocess.feature_store.repository.feature_store_save_repo import (
            FeatureStoreSaveRepository,
        )

        return FeatureStoreSaveRepository(
            config=feature_store_config,
            feature_view_factory=feature_view_factory,
            definition_applier=definition_applier,
            materializer=materializer,
            online_writer=online_feature_writer,
            offline_repo=offline_feature_repository,
            feature_view_name_mapper=feature_view_name_mapper,
        )

    @provider  # type: ignore[misc]
    @singleton
    def provide_feature_store_fetch_repository(
        self,
        online_feature_reader: IOnlineFeatureReader,
        historical_feature_reader: IHistoricalFeatureReader,
    ) -> IFeatureStoreFetchRepository:  # noqa: D401
        from drl_trading_core.preprocess.feature_store.repository.feature_store_fetch_repo import (
            FeatureStoreFetchRepository,
        )

        return FeatureStoreFetchRepository(online_reader=online_feature_reader, historical_reader=historical_feature_reader)

__all__ = ["AdapterModule"]
