
from abc import ABC, abstractmethod

from pandas import DataFrame


class TimescaleRepoInterface(ABC):
    """
    Interface for TimescaleDB repository.
    """

    @abstractmethod
    def store_timeseries_to_db(self, symbol: str, df: DataFrame) -> None:
        """
        Store time series data to the database.

        Args:
            symbol: The symbol for which the data is being stored.
            df: DataFrame containing the time series data.
        """
        pass
