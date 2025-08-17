"""Core port definition for Feast provider.

The concrete implementation lives in the adapter package and must implement
`IFeatureStoreProvider`. Core code depends ONLY on this interface to avoid a
reverse dependency on the adapter layer.
"""
from __future__ import annotations
from abc import ABC
from typing import Protocol, runtime_checkable, Sequence, Iterable, Any

from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo


@runtime_checkable
class FeatureViewLike(Protocol):  # pragma: no cover - structural type
    """Subset of a FeatureView used by core (schema & name)."""

    name: str
    schema: Sequence[Any]  # elements expected to have a 'name' attribute


class FeatureServiceLike(Protocol):  # pragma: no cover - structural type
    """Subset of a FeatureService used by core (name only)."""

    name: str


class EntityLike(Protocol):  # pragma: no cover - structural type
    """Subset of an Entity used by core (name only)."""

    name: str


class FeatureStoreLike(Protocol):  # pragma: no cover - structural type
    """Operations core relies on; implemented by Feast FeatureStore or adapter wrapper."""

    def apply(self, objects: Iterable[Any]) -> None: ...  # noqa: D401,E701
    def materialize(self, start_date, end_date) -> None: ...  # datetime-like params
    def write_to_online_store(self, feature_view_name: str, df) -> None: ...
    def get_online_features(self, features, entity_rows) -> Any: ...  # returns object w/ to_df
    def get_historical_features(self, features, entity_df) -> Any: ...  # returns object w/ to_df
    def get_feature_view(self, name: str) -> FeatureViewLike: ...


class IFeatureStoreProvider(Protocol):  # pragma: no cover - structural type
    """Hexagonal port for feature store operations (infrastructure-agnostic)."""

    def is_enabled(self) -> bool: ...
    def get_feature_store(self) -> FeatureStoreLike: ...
    def create_feature_view(
        self,
        symbol: str,
        feature_view_name: str,
        feature_role: FeatureRoleEnum,
        feature_version_info: FeatureConfigVersionInfo,
    ) -> FeatureViewLike: ...
    def create_feature_service(
        self,
        symbol: str,
        feature_version_info: FeatureConfigVersionInfo,
        feature_views: Sequence[FeatureViewLike] | None = None,
    ) -> FeatureServiceLike: ...
    def get_entity(self, symbol: str) -> EntityLike: ...


class _AdapterNotInstalledFeastProvider(ABC):  # pragma: no cover - failure shim
    """Placeholder implementation raised when adapter not installed."""

    def __init__(self) -> None:
        raise ImportError(
            "Feast adapter implementation not available. Install drl-trading-adapter and bind its AdapterModule."
        )


# Backwards compatible alias for old import path; users should migrate to interface.
FeastProvider = _AdapterNotInstalledFeastProvider  # type: ignore

__all__ = [
    "IFeatureStoreProvider",
    "FeatureStoreLike",
    "FeatureViewLike",
    "FeatureServiceLike",
    "EntityLike",
    "FeastProvider",
]
