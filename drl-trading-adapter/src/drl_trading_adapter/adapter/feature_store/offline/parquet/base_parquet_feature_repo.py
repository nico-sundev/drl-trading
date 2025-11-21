"""
Base implementation for offline feature repositories.

This module provides common business logic for partition-aware feature storage,
with abstract I/O operations that concrete implementations must provide.
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd
from pandas import DataFrame, to_datetime

from drl_trading_adapter.adapter.feature_store.offline.offline_feature_repo_interface import IOfflineFeatureRepository

logger = logging.getLogger(__name__)


class BaseParquetFeatureRepo(IOfflineFeatureRepository):
    """
    Abstract base class providing common partition-aware storage logic.

    This class implements the Template Method pattern:
    - Concrete methods define the business logic (incremental/batch storage)
    - Abstract methods define I/O operations (filesystem vs S3 vs other)

    Subclasses must implement storage-specific I/O operations:
    - _load_metadata / _save_metadata
    - _load_partitions / _delete_partitions
    - _store_with_datetime_organization
    """

    def store_features_incrementally(self, features_df: DataFrame, symbol: str) -> int:
        """
        Store features incrementally using partition-aware deduplication.

        Template method - implements common business logic.
        Subclasses provide I/O operations via abstract methods.

        This method is optimized for daily/periodic feature updates where new data
        may overlap with existing data. It uses partition metadata to avoid
        loading all existing data into memory.

        Args:
            features_df: Features DataFrame with 'event_timestamp' column
            symbol: Symbol identifier for the dataset

        Returns:
            Number of new feature records stored (after deduplication)

        Raises:
            ValueError: If 'event_timestamp' column is missing
        """
        if "event_timestamp" not in features_df.columns:
            raise ValueError("features_df must contain 'event_timestamp' column for incremental storage")

        if features_df.empty:
            logger.info(f"No features to store for {symbol}")
            return 0

        # Prepare features for processing
        features_df = features_df.copy()
        features_df["event_timestamp"] = to_datetime(features_df["event_timestamp"])
        features_df = features_df.drop_duplicates(subset=["event_timestamp"], keep="first")
        features_df = features_df.sort_values("event_timestamp").reset_index(drop=True)

        # Load metadata to find overlapping partitions
        metadata = self._load_metadata(symbol)

        if metadata and metadata.get("partitions"):
            # Find time range of new data
            min_new_time = features_df["event_timestamp"].min()
            max_new_time = features_df["event_timestamp"].max()

            # Find overlapping partitions
            overlapping = self._find_overlapping_partitions(
                metadata["partitions"],
                min_new_time,
                max_new_time
            )

            if overlapping:
                # Load only overlapping partitions for deduplication
                existing_df = self._load_partitions(symbol, overlapping)
                existing_timestamps = set(existing_df["event_timestamp"])
                new_mask = ~features_df["event_timestamp"].isin(existing_timestamps)
                new_features = features_df[new_mask]

                if new_features.empty:
                    logger.info(f"No new features to store for {symbol} (all timestamps already exist)")
                    return 0

                # Combine overlapping existing data with new data
                combined_df = pd.concat([existing_df, new_features], ignore_index=True)
                combined_df = combined_df.sort_values("event_timestamp").reset_index(drop=True)

                # Delete overlapping partitions
                self._delete_partitions(symbol, overlapping)

                # Store combined data and get new partition metadata
                new_partition_metadata = self._store_with_datetime_organization(combined_df, symbol)

                # Update metadata: remove overlapping, add new
                non_overlapping = [p for p in metadata["partitions"] if p not in overlapping]
                metadata["partitions"] = non_overlapping + new_partition_metadata
                self._save_metadata(symbol, metadata)

                logger.info(f"Added {len(new_features)} new feature records for {symbol} (merged with overlapping partitions)")
                return len(new_features)
            else:
                # No overlap - just append new partitions
                new_partition_metadata = self._store_with_datetime_organization(features_df, symbol)
                metadata["partitions"].extend(new_partition_metadata)
                self._save_metadata(symbol, metadata)

                logger.info(f"Added {len(features_df)} new feature records for {symbol} (no overlap)")
                return len(features_df)
        else:
            # First-time storage
            partition_metadata = self._store_with_datetime_organization(features_df, symbol)

            # Create initial metadata
            self._save_metadata(symbol, {"partitions": partition_metadata})

            logger.info(f"Stored {len(features_df)} initial feature records for {symbol}")
            return len(features_df)

    def store_features_batch(self, feature_batches: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Store multiple feature datasets in batch mode with partition-level replacement.

        Template method - implements common business logic.
        Subclasses provide I/O operations via abstract methods.

        Unlike incremental storage, batch mode replaces overlapping partitions entirely
        rather than deduplicating. Only loads and rewrites overlapping partitions,
        keeping non-overlapping data intact.

        Args:
            feature_batches: List of dicts with 'symbol' and 'features_df' keys

        Returns:
            Dict mapping symbol to number of records stored
        """
        results = {}

        for batch in feature_batches:
            symbol = batch["symbol"]
            features_df = batch["features_df"]

            if features_df.empty:
                logger.warning(f"Empty feature DataFrame provided for {symbol}")
                results[symbol] = 0
                continue

            if "event_timestamp" not in features_df.columns:
                raise ValueError("features_df must include 'event_timestamp' column")

            # Deduplicate within the new batch
            features_df = features_df.copy()
            features_df["event_timestamp"] = to_datetime(features_df["event_timestamp"])
            features_df = features_df.drop_duplicates(subset=["event_timestamp"], keep="first")
            features_df = features_df.sort_values("event_timestamp").reset_index(drop=True)

            # Load metadata to find overlapping partitions
            metadata = self._load_metadata(symbol)

            if metadata and metadata.get("partitions"):
                # Find time range of new data
                min_new_time = features_df["event_timestamp"].min()
                max_new_time = features_df["event_timestamp"].max()

                # Find overlapping partitions
                overlapping = self._find_overlapping_partitions(
                    metadata["partitions"],
                    min_new_time,
                    max_new_time
                )

                if overlapping:
                    # Load non-overlapping portion of existing data
                    existing_overlapping = self._load_partitions(symbol, overlapping)

                    # Filter existing data to keep only records outside new range
                    non_overlapping_existing = existing_overlapping[
                        (existing_overlapping["event_timestamp"] < min_new_time) |
                        (existing_overlapping["event_timestamp"] > max_new_time)
                    ]

                    # Combine non-overlapping existing with new data
                    if not non_overlapping_existing.empty:
                        combined_df = pd.concat([non_overlapping_existing, features_df], ignore_index=True)
                        combined_df = combined_df.sort_values("event_timestamp").reset_index(drop=True)
                    else:
                        combined_df = features_df

                    # Delete overlapping partitions
                    self._delete_partitions(symbol, overlapping)

                    # Store combined data
                    new_partition_metadata = self._store_with_datetime_organization(combined_df, symbol)

                    # Update metadata: remove overlapping, add new
                    non_overlapping_partitions = [p for p in metadata["partitions"] if p not in overlapping]
                    metadata["partitions"] = non_overlapping_partitions + new_partition_metadata
                    self._save_metadata(symbol, metadata)

                    logger.info(f"Batch mode: Replaced {len(overlapping)} partition(s) with {len(new_partition_metadata)} new partition(s) for {symbol}")
                else:
                    # No overlap - just append
                    new_partition_metadata = self._store_with_datetime_organization(features_df, symbol)
                    metadata["partitions"].extend(new_partition_metadata)
                    self._save_metadata(symbol, metadata)

                    logger.info("Batch mode: No overlap, appended new partition(s)")

                results[symbol] = len(features_df)
            else:
                # First-time storage
                partition_metadata = self._store_with_datetime_organization(features_df, symbol)
                self._save_metadata(symbol, {"partitions": partition_metadata})

                logger.info("Batch mode: Initial data stored")
                results[symbol] = len(features_df)

        return results

    def _find_overlapping_partitions(
        self,
        partitions: List[Dict[str, Any]],
        min_time: pd.Timestamp,
        max_time: pd.Timestamp
    ) -> List[Dict[str, Any]]:
        """
        Find partitions that overlap with the given time range.

        Concrete method - common logic for all storage backends.

        Args:
            partitions: List of partition metadata dicts
            min_time: Minimum timestamp of new data
            max_time: Maximum timestamp of new data

        Returns:
            List of overlapping partition metadata dicts
        """
        overlapping = []
        for partition in partitions:
            part_min = pd.Timestamp(partition["min_timestamp"])
            part_max = pd.Timestamp(partition["max_timestamp"])

            # Check if ranges overlap: NOT (new_max < part_min OR new_min > part_max)
            if not (max_time < part_min or min_time > part_max):
                overlapping.append(partition)

        return overlapping

    @abstractmethod
    def get_repo_path(self, symbol: str) -> str:
        """Get the repository path for a symbol."""
        pass

    # Abstract I/O operations - subclasses must implement these

    @abstractmethod
    def _load_metadata(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Load partition metadata for a symbol.

        Args:
            symbol: Symbol identifier

        Returns:
            Metadata dict with 'partitions' list, or None if no metadata exists
        """
        pass

    @abstractmethod
    def _save_metadata(self, symbol: str, metadata: Dict[str, Any]) -> None:
        """
        Save partition metadata for a symbol.

        Args:
            symbol: Symbol identifier
            metadata: Metadata dict to save
        """
        pass

    @abstractmethod
    def _load_partitions(self, symbol: str, partitions: List[Dict[str, Any]]) -> DataFrame:
        """
        Load data from specific partitions only.

        Args:
            symbol: Symbol identifier
            partitions: List of partition metadata dicts

        Returns:
            Combined DataFrame from specified partitions
        """
        pass

    @abstractmethod
    def _delete_partitions(self, symbol: str, partitions: List[Dict[str, Any]]) -> None:
        """
        Delete specific partition files/objects.

        Args:
            symbol: Symbol identifier
            partitions: List of partition metadata dicts to delete
        """
        pass

    @abstractmethod
    def _store_with_datetime_organization(self, features_df: DataFrame, symbol: str) -> List[Dict[str, Any]]:
        """
        Store features in datetime-organized partitions.

        Args:
            features_df: Features to store (already deduplicated and sorted)
            symbol: Symbol identifier

        Returns:
            List of partition metadata dicts for newly created partitions
        """
        pass
