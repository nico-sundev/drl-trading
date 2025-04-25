"""Interface definition for feature store services."""

from abc import ABC, abstractmethod
from typing import List, Optional

from pandas import DataFrame


class FeatureStoreInterface(ABC):
    """Interface for feature store services.

    This interface defines the contract for services that handle feature storage,
    retrieval, and management. Implementations can use different backends (Feast,
    custom solutions, etc.) while maintaining a consistent interface.
    """

    @abstractmethod
    def store_features(
        self,
        feature_df: DataFrame,
        feature_name: str,
        param_hash: str,
        sub_feature_names: List[str],
    ) -> None:
        """Store computed features in the feature store.

        Args:
            feature_df: DataFrame containing the computed features
            feature_name: Name of the feature
            param_hash: Hash of the feature parameters
            sub_feature_names: List of sub-feature names in the feature
        """
        pass

    @abstractmethod
    def get_historical_features(
        self, feature_name: str, param_hash: str, sub_feature_names: List[str]
    ) -> Optional[DataFrame]:
        """Retrieve historical features from the feature store.

        Args:
            feature_name: Name of the feature to retrieve
            param_hash: Hash of the feature parameters
            sub_feature_names: List of sub-feature names to retrieve

        Returns:
            DataFrame with historical feature data or None if not found
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the feature store with required infrastructure.

        This method should handle any setup required for the feature store,
        such as creating entities, feature views, or database tables.
        """
        pass

    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if the feature store is enabled.

        Returns:
            True if the feature store is enabled, False otherwise
        """
        pass
