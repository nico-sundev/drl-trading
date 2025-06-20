
from abc import ABC, abstractmethod

from pandas import DataFrame


class ContextFeatureServiceInterface(ABC):
    """
    Interface for services that manage context-related features required by the trading environment.

    This interface defines the contract for preparing, validating, and identifying context features
    that are required for the functioning of the trading environment but are not part of the
    feature set used for machine learning.
    """

    @abstractmethod
    def prepare_context_features(self, base_dataset_source: DataFrame) -> DataFrame:
        """
        Prepares a DataFrame with only the essential context-related features needed by the trading environment.

        This method validates that all required primary columns exist in the dataset,
        computes any necessary derived columns (e.g., ATR), and returns a DataFrame
        containing only the context columns.

        Args:
            base_dataset: The base dataset containing OHLC data

        Returns:
            DataFrame with Time index and required context columns

        Raises:
            ValueError: If required primary columns are missing from the dataset
        """
        pass

    @abstractmethod
    def merge_context_features(
        self, base_df: DataFrame, context_df: DataFrame
    ) -> DataFrame:
        """
        Merges context features into the base DataFrame.

        Args:
            base_df: Base DataFrame with features
            context_df: DataFrame with context features

        Returns:
            DataFrame with both base and context features merged
        """
        pass

    @abstractmethod
    def is_context_column(self, column_name: str) -> bool:
        """
        Checks whether a given column name is considered a context column.

        Args:
            column_name: Name of the column to check

        Returns:
            True if the column is a context column, False otherwise
        """
        pass
