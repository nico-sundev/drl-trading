"""
Local filesystem implementation of offline feature repository.

This module provides feature storage and retrieval using local parquet files
organized in a datetime-based directory structure for efficient access.
"""

import logging
import os
from typing import Optional

import pandas as pd
from drl_trading_common.config.feature_config import FeatureStoreConfig
from drl_trading_common.model.dataset_identifier import DatasetIdentifier
from injector import inject
from pandas import DataFrame, concat, to_datetime

from .offline_feature_repo_interface import OfflineFeatureRepoInterface

logger = logging.getLogger(__name__)


@inject
class OfflineFeatureLocalRepo(OfflineFeatureRepoInterface):
    """
    Local filesystem implementation for offline feature storage.

    Features are stored as parquet files organized in a datetime-based
    directory structure: base_path/symbol/timeframe/year=YYYY/month=MM/day=DD/

    This organization enables:
    - Efficient temporal queries
    - Partitioned storage for large datasets
    - Easy data lifecycle management
    """

    def __init__(self, config: FeatureStoreConfig):
        self.config = config
        self.base_path = config.offline_store_path

    def store_features_incrementally(
        self,
        features_df: DataFrame,
        dataset_id: DatasetIdentifier,
    ) -> int:
        """
        Store features incrementally using datetime-organized directory structure.

        Path structure: base_path/symbol/timeframe/year=YYYY/month=MM/day=DD/features_*.parquet

        Args:
            features_df: Features DataFrame with 'event_timestamp' column
            dataset_id: Dataset identifier

        Returns:
            Number of new feature records stored

        Raises:
            ValueError: If 'event_timestamp' column is missing
        """
        if "event_timestamp" not in features_df.columns:
            raise ValueError("features_df must contain 'event_timestamp' column for incremental storage")

        # Prepare features for processing
        features_df = features_df.copy()
        features_df["event_timestamp"] = to_datetime(features_df["event_timestamp"])
        features_df = features_df.sort_values("event_timestamp").reset_index(drop=True)

        # Load existing features to check for duplicates
        existing_df = self.load_existing_features(dataset_id)

        if existing_df is None or existing_df.empty:
            new_features_df = features_df
            logger.info(f"No existing features found for {dataset_id.symbol}/{dataset_id.timeframe.value}")
        else:
            # Validate schema consistency
            self._validate_schema_consistency(existing_df, features_df, dataset_id)

            # Find new records by timestamp comparison
            existing_timestamps = set(existing_df["event_timestamp"])
            new_mask = ~features_df["event_timestamp"].isin(existing_timestamps)
            new_features_df = features_df[new_mask].copy()

            # Log overlapping timestamps
            overlapping_count = (~new_mask).sum()
            if overlapping_count > 0:
                logger.warning(
                    f"Found {overlapping_count} overlapping timestamps for {dataset_id.symbol}/{dataset_id.timeframe.value}. "
                    f"These will be skipped to prevent data corruption."
                )

        if new_features_df.empty:
            logger.info(f"No new features to store for {dataset_id.symbol}/{dataset_id.timeframe.value}")
            return 0

        # Store new features with datetime organization
        self._store_with_datetime_organization(new_features_df, dataset_id)

        logger.info(
            f"Stored {len(new_features_df)} new features out of {len(features_df)} total "
            f"for {dataset_id.symbol}/{dataset_id.timeframe.value}"
        )

        return len(new_features_df)

    def load_existing_features(self, dataset_id: DatasetIdentifier) -> Optional[DataFrame]:
        """
        Load existing features from all parquet files in the dataset path.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Combined DataFrame of existing features, or None if no files exist
        """
        dataset_path = self._get_dataset_base_path(dataset_id)

        if not os.path.exists(dataset_path):
            return None

        parquet_files = []
        # Recursively find all parquet files
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.parquet'):
                    parquet_files.append(os.path.join(root, file))

        if not parquet_files:
            return None

        # Load and combine all existing feature files
        dfs = []
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
                if "event_timestamp" in df.columns:
                    df["event_timestamp"] = to_datetime(df["event_timestamp"])
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load existing features from {file_path}: {e}")

        if not dfs:
            return None

        # Combine all DataFrames and remove duplicates
        combined_df = concat(dfs, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["event_timestamp"]).sort_values("event_timestamp")

        logger.info(f"Loaded {len(combined_df)} existing feature records from {len(parquet_files)} files")
        return combined_df

    def feature_exists(self, dataset_id: DatasetIdentifier) -> bool:
        """
        Check if features exist for the given dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            True if features exist, False otherwise
        """
        dataset_path = self._get_dataset_base_path(dataset_id)

        if not os.path.exists(dataset_path):
            return False        # Check for any parquet files in the directory tree
        for _, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.parquet'):
                    return True

        return False

    def get_feature_count(self, dataset_id: DatasetIdentifier) -> int:
        """
        Get the total count of feature records for a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Total number of feature records
        """
        existing_df = self.load_existing_features(dataset_id)
        return len(existing_df) if existing_df is not None else 0

    def _get_dataset_base_path(self, dataset_id: DatasetIdentifier) -> str:
        """Get the base path for a specific dataset."""
        return os.path.join(self.base_path, dataset_id.symbol, dataset_id.timeframe.value)

    def _validate_schema_consistency(
        self,
        existing_df: DataFrame,
        new_df: DataFrame,
        dataset_id: DatasetIdentifier
    ) -> None:
        """
        Validate that new features have consistent schema with existing features.

        Args:
            existing_df: Existing features DataFrame
            new_df: New features DataFrame
            dataset_id: Dataset identifier for error context

        Raises:
            ValueError: If schema inconsistencies are detected
        """
        existing_cols = set(existing_df.columns)
        new_cols = set(new_df.columns)

        # Check for missing columns in new data
        missing_cols = existing_cols - new_cols
        if missing_cols:
            raise ValueError(
                f"Schema validation failed for {dataset_id.symbol}/{dataset_id.timeframe.value}: "
                f"New features missing columns: {sorted(missing_cols)}"
            )

        # Allow new columns but log them
        extra_cols = new_cols - existing_cols
        if extra_cols:
            logger.info(
                f"New feature columns detected for {dataset_id.symbol}/{dataset_id.timeframe.value}: "
                f"{sorted(extra_cols)}"
            )

        # Validate common column types
        for col in existing_cols.intersection(new_cols):
            if col == "event_timestamp":  # Skip timestamp as we normalize it
                continue

            existing_dtype = existing_df[col].dtype
            new_dtype = new_df[col].dtype

            # Allow compatible numeric types
            if pd.api.types.is_numeric_dtype(existing_dtype) and pd.api.types.is_numeric_dtype(new_dtype):
                continue

            if existing_dtype != new_dtype:
                logger.warning(
                    f"Column type mismatch for '{col}' in {dataset_id.symbol}/{dataset_id.timeframe.value}: "
                    f"existing={existing_dtype}, new={new_dtype}"
                )

    def _store_with_datetime_organization(
        self,
        features_df: DataFrame,
        dataset_id: DatasetIdentifier
    ) -> None:
        """
        Store features using datetime-based path organization.

        Path structure: base_path/symbol/timeframe/year=YYYY/month=MM/day=DD/features_*.parquet

        Args:
            features_df: Features DataFrame to store
            dataset_id: Dataset identifier
        """
        if features_df.empty:
            return

        base_path = self._get_dataset_base_path(dataset_id)

        # Group features by date for organized storage
        features_df["_date"] = features_df["event_timestamp"].dt.date

        for date_key, group_df in features_df.groupby("_date"):
            # Ensure date_key is a date object
            if hasattr(date_key, 'year'):
                year = date_key.year
                month = date_key.month
                day = date_key.day
            else:
                # Fallback for edge cases
                date_obj = pd.to_datetime(str(date_key)).date()
                year = date_obj.year
                month = date_obj.month
                day = date_obj.day

            # Create datetime-organized directory structure
            date_path = os.path.join(
                base_path,
                f"year={year}",
                f"month={month:02d}",
                f"day={day:02d}"
            )
            os.makedirs(date_path, exist_ok=True)

            # Generate timestamped filename
            min_time = group_df["event_timestamp"].min()
            max_time = group_df["event_timestamp"].max()
            filename = f"features_{min_time.strftime('%Y%m%d_%H%M%S')}_{max_time.strftime('%H%M%S')}.parquet"

            file_path = os.path.join(date_path, filename)

            # Remove temporary date column before storage
            store_df = group_df.drop(columns=["_date"])

            # Store with optimal parquet settings
            store_df.to_parquet(
                file_path,
                index=False,
                compression="snappy",  # Good balance of speed and compression
                engine="pyarrow"  # Fast and reliable
            )

            logger.info(
                f"Stored {len(store_df)} features for {year}-{month:02d}-{day:02d} in {file_path} "
                f"({dataset_id.symbol}/{dataset_id.timeframe.value})"
            )
