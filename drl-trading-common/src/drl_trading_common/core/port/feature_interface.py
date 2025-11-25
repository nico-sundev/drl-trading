from abc import ABC, abstractmethod

from drl_trading_common.core.model.feature_metadata import FeatureMetadata


class IFeature(ABC):
    """Abstract interface for features, used only in tests (adapters depend on ports, not domain models)."""

    @abstractmethod
    def _get_sub_features_names(self) -> list[str]:
        pass

    @abstractmethod
    def get_metadata(self) -> FeatureMetadata:
        pass
