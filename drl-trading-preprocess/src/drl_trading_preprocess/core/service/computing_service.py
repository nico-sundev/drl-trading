
import logging
from abc import ABC, abstractmethod

from injector import inject
from pandas import DataFrame, Series

from drl_trading_core.preprocess.feature.feature_manager import (
    FeatureManager,
)

logger = logging.getLogger(__name__)

class IFeatureComputer(ABC):
    @abstractmethod
    def compute_batch(self, data: DataFrame) -> DataFrame:
        """Compute results on a batch of data."""
        pass

    @abstractmethod
    def compute_incremental(self, data_point: Series) -> Series:
        """Compute results incrementally for a single data point."""
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the computing service with initial data."""
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
        self._initialized = False

    def initialize(self) -> bool:
        """
        Initialize the computing service with initial data.

        Args:
            initial_data: Initial data to initialize features with.

        Returns:
            True if initialization was successful, False otherwise.
        """
        try:
            # Initialize feature manager with initial data
            self.feature_manager_service.initialize_features()
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize computing service: {str(e)}")
            return False

    def compute_batch(self, data: DataFrame) -> DataFrame:
        """
        Compute results on a batch of data.

        Args:
            data: Data to compute features for.

        Returns:
            DataFrame with computed features.
        """
        if not self._initialized:
            logger.warning("ComputingService not initialized. Initializing with provided data.")
            self.initialize()

        # Update features with new data
        self.feature_manager_service.add(data)

        # Compute all features
        result = self.feature_manager_service.compute_all()
        if result is None:
            logger.warning("No results computed. Returning empty DataFrame.")
            return DataFrame()

        return result

    def compute_incremental(self, data_point: Series) -> Series:
        """
        Compute results incrementally for a single data point.

        Args:
            data_point: A single data point to compute features for.

        Returns:
            Series with computed features for the data point.
        """
        if not self._initialized:
            logger.error("ComputingService not initialized. Cannot compute incremental data.")
            return Series()

        # Convert Series to DataFrame with a single row
        data_df = DataFrame([data_point])

        # Update features with new data point
        self.feature_manager_service.add(data_df)

        # Compute latest features
        result_df = self.feature_manager_service.compute_latest()

        if result_df is None or result_df.empty:
            logger.warning("No incremental results computed. Returning empty Series.")
            return Series()

        # Convert the result DataFrame's last row to a Series
        return result_df.iloc[-1] if not result_df.empty else Series()
