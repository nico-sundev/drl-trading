
from abc import ABC, abstractmethod
from typing import Optional

from pandas import DataFrame


class TimescaleRepoInterface(ABC):
    """
    Interface for TimescaleDB repository operations.

    Defines the contract for storing and retrieving time series market data
    in TimescaleDB with proper abstraction for testability.
    """

    @abstractmethod
    def save_market_data(self, symbol: str, timeframe: str, df: DataFrame) -> None:
        """
        Store time series data to the database.

        Args:
            symbol: The trading symbol (e.g., "EURUSD")
            timeframe: The timeframe (e.g., "1H", "1D", "5M")
            df: DataFrame containing the time series OHLCV data

        Raises:
            ValueError: If required columns are missing from DataFrame
            DatabaseConnectionError: If database operation fails
        """
        pass

    @abstractmethod
    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[str]:
        """
        Get the latest timestamp for a symbol/timeframe combination.

        Args:
            symbol: The trading symbol
            timeframe: The timeframe

        Returns:
            str: Latest timestamp as ISO string, or None if no data exists
        """
        pass
