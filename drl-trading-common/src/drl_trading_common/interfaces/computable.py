from abc import ABC, abstractmethod
from typing import Optional

from pandas import DataFrame


class Computable(ABC):

    @abstractmethod
    def compute_all(self) -> Optional[DataFrame]:
        pass

    @abstractmethod
    def add(self, df: DataFrame) -> None:
        """Add new data to the feature. This method should be implemented by subclasses to handle new data."""
        pass

    @abstractmethod
    def compute_latest(self) -> Optional[DataFrame]:
        pass
