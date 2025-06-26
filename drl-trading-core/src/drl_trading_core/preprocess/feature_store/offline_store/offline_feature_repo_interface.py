"""
Interface for offline feature storage repositories.

This module defines the contract for storing and retrieving features
from different storage backends (local filesystem, S3, etc.).
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from pandas import DataFrame

logger = logging.getLogger(__name__)


class OfflineFeatureRepoInterface(ABC):
    """
    Interface for offline feature storage operations.

    This interface abstracts the storage backend, allowing implementations
    for local filesystem, S3, or other storage systems while maintaining
    the same API for feature storage and retrieval.
    """

    @abstractmethod
    def store_features_incrementally(
        self,
        features_df: DataFrame,
        symbol: str,
    ) -> int:
        """
        Store features incrementally, avoiding duplicates and optimizing storage.

        Args:
            features_df: DataFrame containing features with 'event_timestamp' column
            dataset_id: Identifier for the dataset

        Returns:
            Number of new feature records stored

        Raises:
            ValueError: If features_df lacks required 'event_timestamp' column
        """
        pass

    @abstractmethod
    def load_existing_features(
        self,
        symbol: str,
    ) -> Optional[DataFrame]:
        """
        Load all existing features for a dataset.

        Args:
            dataset_id: Identifier for the dataset

        Returns:
            Combined DataFrame of existing features, or None if no features exist
        """
        pass

    @abstractmethod
    def feature_exists(
        self,
        symbol: str,
    ) -> bool:
        """
        Check if features exist for the given dataset.

        Args:
            dataset_id: Identifier for the dataset

        Returns:
            True if features exist, False otherwise
        """
        pass

    @abstractmethod
    def get_feature_count(
        self,
        symbol: str,
    ) -> int:
        """
        Get the total count of feature records for a dataset.

        Args:
            dataset_id: Identifier for the dataset

        Returns:
            Total number of feature records
        """
        pass
