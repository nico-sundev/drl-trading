"""Hexagonal port definitions for feature store provider and handles.

These Protocols form the stable boundary the core depends on. Concrete
implementations live in adapter packages (e.g., Feast) and are wired via DI.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.model.feature_config_version_info import (
    FeatureConfigVersionInfo,
)


@runtime_checkable
class FeatureViewDescriptor(Protocol):  # pragma: no cover - structural type
    """Subset of a FeatureView used by core (schema & name)."""

    name: str
    schema: Sequence[Any]  # elements expected to have a 'name' attribute


class FeatureServiceHandle(Protocol):  # pragma: no cover - structural type
    """Subset of a FeatureService used by core (name only)."""

    name: str


class EntityHandle(Protocol):  # pragma: no cover - structural type
    """Subset of an Entity used by core (name only)."""

    name: str


class FeatureStoreClient(Protocol):  # pragma: no cover - structural type
    """Operations core relies on; implemented by Feast FeatureStore or adapter wrapper."""

    def apply(self, objects: Iterable[Any]) -> None: ...  # noqa: D401,E701

    def materialize(self, start_date, end_date) -> None: ...  # datetime-like params

    def write_to_online_store(self, feature_view_name: str, df) -> None: ...

    def get_online_features(self, features, entity_rows) -> Any: ...  # returns object w/ to_df

    def get_historical_features(self, features, entity_df) -> Any: ...  # returns object w/ to_df

    def get_feature_view(self, name: str) -> FeatureViewDescriptor: ...


class IFeatureStoreProvider(Protocol):  # pragma: no cover - structural type
    """Hexagonal port for feature store operations (infrastructure-agnostic)."""

    def is_enabled(self) -> bool: ...

    def get_feature_store(self) -> FeatureStoreClient: ...

    def create_feature_view(
        self,
        symbol: str,
        feature_view_name: str,
        feature_role: FeatureRoleEnum,
        feature_version_info: FeatureConfigVersionInfo,
    ) -> FeatureViewDescriptor: ...

    def create_feature_service(
        self,
        symbol: str,
        feature_version_info: FeatureConfigVersionInfo,
        feature_views: Sequence[FeatureViewDescriptor] | None = None,
    ) -> FeatureServiceHandle: ...

    def get_entity(self, symbol: str) -> EntityHandle: ...


__all__ = [
    "IFeatureStoreProvider",
    "FeatureStoreClient",
    "FeatureViewDescriptor",
    "FeatureServiceHandle",
    "EntityHandle",
]
