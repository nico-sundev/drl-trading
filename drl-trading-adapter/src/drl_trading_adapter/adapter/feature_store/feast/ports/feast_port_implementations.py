"""Feast-backed implementations of granular feature store ports.

Bridges core port abstractions (FeatureViewDef, FeatureServiceDef, ...) to concrete
Feast objects while keeping Feast types invisible to core layer.
"""
from __future__ import annotations

from typing import Sequence
import logging

import pandas as pd
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo

from drl_trading_core.preprocess.feature_store.port.feature_store_operation_ports import (
    FeatureViewDef,
    FeatureServiceDef,
    EntityDef,
    IFeatureViewFactory,
    IFeatureDefinitionApplier,
    IFeatureMaterializer,
    IOnlineFeatureWriter,
    IOnlineFeatureReader,
    IHistoricalFeatureReader,
)
from drl_trading_core.preprocess.feature_store.provider.feast_provider import IFeatureStoreProvider

logger = logging.getLogger(__name__)

# NOTE: Implementation uses existing FeastProvider to avoid code duplication. In a later
# refactor provider responsibilities can be split or deprecated.

class FeastFeatureViewFactory(IFeatureViewFactory):
    def __init__(self, feast_provider: IFeatureStoreProvider):
        self._provider = feast_provider

    def create_feature_view(self, symbol: str, feature_view_name: str, feature_role: FeatureRoleEnum, feature_version_info: FeatureConfigVersionInfo) -> FeatureViewDef:  # type: ignore[override]
        fv = self._provider.create_feature_view(symbol, feature_view_name, feature_role, feature_version_info)
        # Extract schema field names
        schema_fields = [f.name for f in getattr(fv, 'schema', [])]
        return FeatureViewDef(name=fv.name, schema_field_names=schema_fields)

    def create_feature_service(self, symbol: str, feature_version_info: FeatureConfigVersionInfo, feature_views: Sequence[FeatureViewDef]) -> FeatureServiceDef:  # type: ignore[override]
        # Reconstruct actual Feast feature views from registry by name (they should have been applied already)
        feast_views = []
        store = self._provider.get_feature_store()
        for v in feature_views:
            try:
                feast_views.append(store.get_feature_view(v.name))
            except Exception:  # pragma: no cover - fallback path
                logger.debug("Feature view %s not yet in registry; skipping add to service", v.name)
        svc = self._provider.create_feature_service(symbol=symbol, feature_version_info=feature_version_info, feature_views=feast_views)
        return FeatureServiceDef(name=svc.name)

    def get_entity(self, symbol: str) -> EntityDef:  # type: ignore[override]
        ent = self._provider.get_entity(symbol)
        return EntityDef(ent.name)

    def get_feature_view_schema(self, feature_view_name: str) -> list[str]:  # type: ignore[override]
        try:
            fv = self._provider.get_feature_store().get_feature_view(feature_view_name)
            return [f.name for f in getattr(fv, 'schema', [])]
        except Exception as e:  # pragma: no cover
            logger.debug("Schema lookup failed for feature view %s: %s", feature_view_name, e)
            return []


class FeastFeatureDefinitionApplier(IFeatureDefinitionApplier):  # pragma: no cover - thin wrapper
    def __init__(self, feast_provider: IFeatureStoreProvider):
        self._provider = feast_provider
        self._fs = feast_provider.get_feature_store()

    def apply_definitions(self, entity: EntityDef, feature_views: Sequence[FeatureViewDef], feature_service: FeatureServiceDef) -> None:  # type: ignore[override]
        objs = []
        # Rehydrate Feast objects by name
        try:
            objs.append(self._fs.get_entity(entity.name))
        except Exception as e:  # pragma: no cover
            logger.warning("Entity %s not found in registry prior to apply: %s", entity.name, e)
        for fv in feature_views:
            try:
                objs.append(self._fs.get_feature_view(fv.name))
            except Exception as e:  # pragma: no cover
                logger.warning("Feature view %s not found during apply: %s", fv.name, e)
        try:
            objs.append(self._fs.get_feature_service(feature_service.name))
        except Exception as e:  # pragma: no cover
            logger.warning("Feature service %s not found during apply: %s", feature_service.name, e)
        if objs:
            self._fs.apply(objs)


class FeastFeatureMaterializer(IFeatureMaterializer):  # pragma: no cover - passthrough
    def __init__(self, feast_provider: IFeatureStoreProvider):
        self._fs = feast_provider.get_feature_store()

    def materialize(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> None:  # type: ignore[override]
        self._fs.materialize(start_ts=start_ts, end_ts=end_ts)


class FeastOnlineFeatureWriter(IOnlineFeatureWriter):  # pragma: no cover - passthrough
    def __init__(self, feast_provider: IFeatureStoreProvider):
        self._fs = feast_provider.get_feature_store()

    def write(self, feature_view_name: str, df: pd.DataFrame) -> None:  # type: ignore[override]
        self._fs.write_to_online_store(feature_view_name=feature_view_name, df=df)


class FeastOnlineFeatureReader(IOnlineFeatureReader):  # pragma: no cover - thin wrapper
    def __init__(self, feast_provider: IFeatureStoreProvider):
        self._provider = feast_provider
        self._fs = feast_provider.get_feature_store()

    def get_online(self, symbol: str, feature_version_info: FeatureConfigVersionInfo) -> pd.DataFrame:  # type: ignore[override]
        svc = self._provider.create_feature_service(symbol=symbol, feature_version_info=feature_version_info)
        entity_rows = [{"symbol": symbol}]
        return self._fs.get_online_features(features=svc, entity_rows=entity_rows).to_df()


class FeastHistoricalFeatureReader(IHistoricalFeatureReader):  # pragma: no cover - thin wrapper
    def __init__(self, feast_provider: IFeatureStoreProvider):
        self._provider = feast_provider
        self._fs = feast_provider.get_feature_store()

    def get_offline(self, symbol: str, timestamps: pd.Series, feature_version_info: FeatureConfigVersionInfo) -> pd.DataFrame:  # type: ignore[override]
        if timestamps.isnull().any():
            timestamps = timestamps.dropna()
        if timestamps.empty:
            return pd.DataFrame()
        if hasattr(timestamps, 'dt') and timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize('UTC')
        entity_df = pd.DataFrame({"symbol": symbol, "event_timestamp": timestamps})
        svc = self._provider.create_feature_service(symbol=symbol, feature_version_info=feature_version_info)
        return self._fs.get_historical_features(features=svc, entity_df=entity_df).to_df()

__all__ = [
    "FeastFeatureViewFactory",
    "FeastFeatureDefinitionApplier",
    "FeastFeatureMaterializer",
    "FeastOnlineFeatureWriter",
    "FeastOnlineFeatureReader",
    "FeastHistoricalFeatureReader",
]
