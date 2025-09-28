from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from pandas import DataFrame


class Computable(ABC):

    @abstractmethod
    def compute_all(self) -> Optional[DataFrame]:
        pass

    @abstractmethod
    def update(self, df: DataFrame) -> None:
        """Add new data to the feature. This method should be implemented by subclasses to handle new data."""
        pass

    @abstractmethod
    def compute_latest(self) -> Optional[DataFrame]:
        pass

    @abstractmethod
    def are_features_caught_up(self, reference_time: datetime) -> bool:
        """
        Check if the feature is caught up based on the last available record time.

        Args:
            reference_time: The current or target datetime to compare against

        Returns:
            True if the feature is caught up (time difference < timeframe duration), False otherwise
        """
        pass
