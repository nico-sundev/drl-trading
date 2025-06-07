
from abc import ABC, abstractmethod
from typing import Optional

from pandas import DataFrame


class BaseIndicator(ABC):

    @abstractmethod
    def add(self, value: DataFrame) -> None:
        """
        Incrementally compute the indicator with a new value.

        :param value: New value to update the indicator with.
        """
        pass

    @abstractmethod
    def get_all(self) -> Optional[DataFrame]:
        """
        Compute the indicator for a batch of data.

        :param data: Data to compute the indicator on.
        :return: Computed indicator values.
        """
        pass

    @abstractmethod
    def get_latest(self) -> Optional[DataFrame]:
        """
        Get the latest computed value of the indicator.

        :return: Latest indicator value.
        """
        pass
