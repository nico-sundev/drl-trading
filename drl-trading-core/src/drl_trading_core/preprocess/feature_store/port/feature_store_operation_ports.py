"""Hexagonal port abstractions for feature store operations.

These fine-grained interfaces decouple core business orchestration from any
specific feature store (e.g., Feast). Adapter layer provides concrete
implementations. Repositories and services should depend on these, not on
provider or underlying client objects.
"""
from __future__ import annotations

from typing import Protocol, Sequence
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo
import pandas as pd


class FeatureViewDef:
    """Lightweight definition / handle returned by IFeatureViewFactory."""

    def __init__(self, name: str, schema_field_names: Sequence[str]):
        self.name = name
        self.schema_field_names = list(schema_field_names)


class FeatureServiceDef:
    """Lightweight feature service handle (name only)."""

    def __init__(self, name: str):
        self.name = name


class EntityDef:
    """Lightweight entity handle (name only)."""

    def __init__(self, name: str):
        self.name = name


class IFeatureViewFactory(Protocol):  # pragma: no cover - structural
    def create_feature_view(
        self,
        symbol: str,
        feature_view_name: str,
        feature_role: FeatureRoleEnum,
        feature_version_info: FeatureConfigVersionInfo,
    ) -> FeatureViewDef: ...  # noqa: D401,E701

    def create_feature_service(
        self,
        symbol: str,
        feature_version_info: FeatureConfigVersionInfo,
        feature_views: Sequence[FeatureViewDef],
    ) -> FeatureServiceDef: ...

    def get_entity(self, symbol: str) -> EntityDef: ...

    def get_feature_view_schema(self, feature_view_name: str) -> list[str]: ...  # Field names only


class IFeatureDefinitionApplier(Protocol):  # pragma: no cover
    def apply_definitions(
        self,
        entity: EntityDef,
        feature_views: Sequence[FeatureViewDef],
        feature_service: FeatureServiceDef,
    ) -> None: ...


class IFeatureMaterializer(Protocol):  # pragma: no cover
    def materialize(self, start_ts, end_ts) -> None: ...  # datetime-like parameters


class IOnlineFeatureWriter(Protocol):  # pragma: no cover
    def write(self, feature_view_name: str, df) -> None: ...


class IOnlineFeatureReader(Protocol):  # pragma: no cover
    def get_online(
        self, symbol: str, feature_version_info: FeatureConfigVersionInfo
    ) -> pd.DataFrame: ...


class IHistoricalFeatureReader(Protocol):  # pragma: no cover
    def get_offline(
        self,
        symbol: str,
        timestamps: pd.Series,
        feature_version_info: FeatureConfigVersionInfo,
    ) -> pd.DataFrame: ...


__all__ = [
    "FeatureViewDef",
    "FeatureServiceDef",
    "EntityDef",
    "IFeatureViewFactory",
    "IFeatureDefinitionApplier",
    "IFeatureMaterializer",
    "IOnlineFeatureWriter",
    "IOnlineFeatureReader",
    "IHistoricalFeatureReader",
]
