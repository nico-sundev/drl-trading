"""
Interface for offline feature storage repositories.

This module defines the contract for storing and retrieving features
from different storage backends (local filesystem, S3, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pandas import DataFrame

class IOfflineFeatureRepository(ABC):
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
            symbol: Symbol identifier for the dataset

        Returns:
            Number of new feature records stored

        Raises:
            ValueError: If features_df lacks required 'event_timestamp' column
            StorageException: For backend-specific storage failures
        """
        pass

    @abstractmethod
    def store_features_batch(
        self,
        feature_batches: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Store multiple feature datasets in a batch operation.

        Args:
            feature_batches: List of dicts with 'features_df' and 'symbol' keys

        Returns:
            Dict mapping symbol to number of records stored

        Raises:
            StorageException: For batch operation failures
        """
        pass

    @abstractmethod
    def get_repo_path(self, symbol: str) -> str:
        """
        Get the repository path for storing features for a given symbol.

        This method abstracts the path construction logic from the caller,
        allowing different repository implementations to handle path generation
        according to their specific requirements (local filesystem, S3, etc.).

        Args:
            symbol: Symbol identifier for the dataset

        Returns:
            str: The repository path where features for this symbol should be stored

        Raises:
            ValueError: If symbol is invalid
            StorageException: For backend-specific path resolution failures
        """
        pass
