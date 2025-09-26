
import logging
from abc import ABC, abstractmethod

from injector import inject
from pandas import DataFrame, Series

from drl_trading_common.config.feature_config import FeaturesConfig
from drl_trading_core.core.service.feature_manager import (
    FeatureManager,
)

logger = logging.getLogger(__name__)

class IFeatureComputer(ABC):
    @abstractmethod
    def compute_batch(self, data: DataFrame, features_config: FeaturesConfig) -> DataFrame:
        """Compute results on a batch of data."""
        pass

    @abstractmethod
    def compute_incremental(self, data_point: Series, features_config: FeaturesConfig) -> Series:
        """Compute results incrementally for a single data point."""
        pass

class FeatureComputingService(IFeatureComputer):

    @inject
    def __init__(
        self,
        feature_manager_service: FeatureManager
    ) -> None:
        """
        Initialize the ComputingService with configuration and services.

        Args:
            feature_manager_service: Service that manages feature instances.
        """
        self.feature_manager_service = feature_manager_service

    def compute_batch(self, data: DataFrame, features_config: FeaturesConfig) -> DataFrame:
        """
        Compute results on a batch of data.

        Args:
            data: Data to compute features for.
            features_config: Configuration for the features to compute.

        Returns:
            DataFrame with computed features.
        """

        # Update features with new data
        self.feature_manager_service.request_features_update(data, features_config)

        # Compute all features
        result = self.feature_manager_service.compute_all()
        if result is None:
            logger.warning("No results computed. Returning empty DataFrame.")
            return DataFrame()

        return result

    def compute_incremental(self, data_point: Series, features_config: FeaturesConfig) -> Series:
        """
        Compute results incrementally for a single data point.

        Args:
            data_point: A single data point to compute features for.

        Returns:
            Series with computed features for the data point.
        """

        # Convert Series to DataFrame with a single row
        data_df = DataFrame([data_point])

        # Update features with new data point
        self.feature_manager_service.request_features_update(data_df, features_config)

        # Compute latest features
        result_df = self.feature_manager_service.compute_latest()

        if result_df is None or result_df.empty:
            logger.warning("No incremental results computed. Returning empty Series.")
            return Series()

        # Convert the result DataFrame's last row to a Series
        return result_df.iloc[-1] if not result_df.empty else Series()
