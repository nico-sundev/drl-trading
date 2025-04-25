"""Interface definition for feature computation services."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pandas import DataFrame

from ai_trading.model.asset_price_dataset import AssetPriceDataSet


class FeatureServiceInterface(ABC):
    """Interface for feature computation services.

    This interface defines the contract for services that compute trading features.
    Implementations can use different strategies for feature computation, caching,
    and optimization while presenting a unified interface.
    """

    @abstractmethod
    def compute_features(
        self,
        asset_data: AssetPriceDataSet,
        symbol: str,
        feature_names: Optional[List[str]] = None,
    ) -> DataFrame:
        """Compute features for the given asset price dataset.

        Args:
            asset_data: The asset price dataset to compute features for
            symbol: The trading symbol being processed
            feature_names: Optional list of specific feature names to compute
                           (None means compute all configured features)

        Returns:
            DataFrame containing the original data with computed features added as columns
        """
        pass

    @abstractmethod
    def get_cached_features(
        self, feature_name: str, params: Dict[str, Any], symbol: str
    ) -> Optional[DataFrame]:
        """Retrieve previously computed features from cache if available.

        Args:
            feature_name: Name of the feature to retrieve
            params: Parameters used to compute the feature
            symbol: The trading symbol

        Returns:
            DataFrame with cached feature data or None if not found
        """
        pass
