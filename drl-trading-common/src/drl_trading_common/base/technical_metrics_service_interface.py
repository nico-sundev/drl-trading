
from abc import ABC, abstractmethod
from typing import Optional

from pandas import DataFrame


class TechnicalMetricsServiceInterface(ABC):
    """
    Interface defining the contract for technical metrics service operations.

    Implementations of this interface are responsible for:
    1. Computing common technical indicators/metrics that may be used by multiple features
    2. Providing efficient access to these metrics to avoid redundant calculations
    3. Supporting various timeframes through separate instances
    """

    @abstractmethod
    def get_atr(self, period: int = 14) -> Optional[DataFrame]:
        """
        Get Average True Range (ATR) values for the current timeframe.

        Args:
            period: The period for ATR calculation, default is 14

        Returns:
            DataFrame: DataFrame with DatetimeIndex and ATR values, or None if calculation fails
        """
        pass

    @property
    @abstractmethod
    def timeframe(self) -> str:
        """
        Get the timeframe this metrics service is associated with.

        Returns:
            str: The timeframe identifier (e.g., "H1", "D1")
        """
        pass
