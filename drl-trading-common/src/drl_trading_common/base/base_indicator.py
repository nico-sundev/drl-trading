
from abc import ABC, abstractmethod
from typing import Optional

from pandas import DataFrame


class BaseIndicator(ABC):
    """
    Base class for technical indicators with timestamp preservation.

    This class provides common functionality for storing timestamps alongside
    indicator values, ensuring proper temporal alignment for feature store operations.
    """

    def __init__(self) -> None:
        """Initialize base indicator with empty timestamp storage."""
        self.timestamps: list = []  # Store timestamps to align with indicator values

    def _store_timestamps(self, df: DataFrame) -> None:
        """
        Extract and store timestamps from DataFrame index.

        Timestamps are accumulated across multiple add() calls to maintain
        temporal alignment with indicator values computed incrementally.

        Args:
            df: DataFrame with DatetimeIndex containing timestamps to store
        """
        if hasattr(df.index, 'to_pydatetime'):
            self.timestamps.extend(df.index.to_pydatetime())
        else:
            self.timestamps.extend(df.index.tolist())

    def _create_result_dataframe(self, values: list, column_name: str) -> DataFrame:
        """
        Create DataFrame with proper DatetimeIndex aligned to indicator values.

        Since indicators may have warmup periods (e.g., RSI needs 14 periods),
        the number of computed values may be less than stored timestamps.
        This method aligns timestamps with available indicator values.

        Args:
            values: List of computed indicator values
            column_name: Name for the result column

        Returns:
            DataFrame with DatetimeIndex and indicator values
        """
        return DataFrame(
            {column_name: values},
            index=self.timestamps[:len(values)]
        )

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
