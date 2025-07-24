"""
Local filesystem implementation of offline feature repository.

This module provides feature storage and retrieval using local parquet files
organized in a datetime-based directory structure for efficient access.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from drl_trading_common.config.feature_config import FeatureStoreConfig
from injector import inject
from pandas import DataFrame, concat, to_datetime

from .offline_feature_repo_interface import IOfflineFeatureRepository

logger = logging.getLogger(__name__)


@inject
class OfflineFeatureLocalRepo(IOfflineFeatureRepository):
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
        self.base_path = config.repo_path

    def store_features_incrementally(
        self,
        features_df: DataFrame,
        symbol: str,
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

        # Deduplicate within the incoming batch first
        features_df = features_df.drop_duplicates(subset=["event_timestamp"]).reset_index(drop=True)

        # Load existing features to check for duplicates
        existing_df = self.load_existing_features(symbol)

        if existing_df is None or existing_df.empty:
            new_features_df = features_df
            logger.info(f"No existing features found for {symbol}")
        else:
            # Validate schema consistency
            self._validate_schema_consistency(existing_df, features_df, symbol)

            # Find new records by timestamp comparison
            existing_timestamps = set(existing_df["event_timestamp"])
            new_mask = ~features_df["event_timestamp"].isin(existing_timestamps)
            new_features_df = features_df[new_mask].copy()

            # Log overlapping timestamps
            overlapping_count = (~new_mask).sum()
            if overlapping_count > 0:
                logger.warning(
                    f"Found {overlapping_count} overlapping timestamps for {symbol}. "
                    f"These will be skipped to prevent data corruption."
                )

        if new_features_df.empty:
            logger.info(f"No new features to store for {symbol}")
            return 0

        # Store new features with datetime organization
        self._store_with_datetime_organization(new_features_df, symbol)

        logger.info(
            f"Stored {len(new_features_df)} new features out of {len(features_df)} total "
            f"for {symbol}"
        )

        return len(new_features_df)

    def load_existing_features(self, symbol: str) -> Optional[DataFrame]:
        """
        Load existing features from all parquet files in the dataset path.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Combined DataFrame of existing features, or None if no files exist
        """
        dataset_path = self._get_dataset_base_path(symbol)

        parquet_files = []
        # Recursively find all parquet files
        try:
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith('.parquet'):
                        parquet_files.append(os.path.join(root, file))
        except OSError:
            # Directory doesn't exist or can't be accessed
            return None

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

    def feature_exists(self, symbol: str) -> bool:
        """
        Check if features exist for the given dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            True if features exist, False otherwise
        """
        dataset_path = self._get_dataset_base_path(symbol)

        if not os.path.exists(dataset_path):
            return False

        # Check for any parquet files in the directory tree
        for _, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.parquet'):
                    return True

        return False

    def get_feature_count(self, symbol: str) -> int:
        """
        Get the total count of feature records for a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Total number of feature records
        """
        existing_df = self.load_existing_features(symbol)
        return len(existing_df) if existing_df is not None else 0

    def _get_dataset_base_path(self, symbol: str) -> str:
        """Get the base path for a specific dataset."""
        return os.path.join(self.base_path, symbol)

    def _validate_schema_consistency(
        self,
        existing_df: DataFrame,
        new_df: DataFrame,
        symbol: str
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
                f"Schema mismatch for {symbol}: "
                f"New features missing columns: {sorted(missing_cols)}"
            )

        # Allow new columns but log them
        extra_cols = new_cols - existing_cols
        if extra_cols:
            logger.info(
                f"New feature columns detected for {symbol}: "
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
                    f"Column type mismatch for '{col}' in {symbol}: "
                    f"existing={existing_dtype}, new={new_dtype}"
                )

    def _store_with_datetime_organization(
        self,
        features_df: DataFrame,
        symbol: str
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

        base_path = self._get_dataset_base_path(symbol)

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
                f"({symbol})"
            )

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
            ValueError: For invalid batch data structure
        """
        results = {}

        for batch in feature_batches:
            if 'features_df' not in batch or 'symbol' not in batch:
                raise ValueError("Each batch must contain 'features_df' and 'symbol' keys")

            symbol = batch['symbol']
            features_df = batch['features_df']

            try:
                stored_count = self.store_features_incrementally(features_df, symbol)
                results[symbol] = stored_count
                logger.info(f"Batch stored {stored_count} features for {symbol}")
            except Exception as e:
                logger.error(f"Failed to store batch for {symbol}: {e}")
                results[symbol] = 0

        return results

    def delete_features(
        self,
        symbol: str,
    ) -> bool:
        """
        Delete all features for a given symbol.

        Args:
            symbol: Symbol identifier for the dataset

        Returns:
            True if deletion was successful, False if symbol didn't exist

        Raises:
            OSError: For filesystem operation failures
        """
        import shutil

        dataset_path = self._get_dataset_base_path(symbol)

        if not os.path.exists(dataset_path):
            logger.info(f"No features found to delete for {symbol}")
            return False

        try:
            shutil.rmtree(dataset_path)
            logger.info(f"Successfully deleted all features for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete features for {symbol}: {e}")
            raise

    def get_storage_metrics(
        self,
        symbol: str,
    ) -> Dict[str, Any]:
        """
        Get storage metrics for a dataset (size, object count, etc.).

        Args:
            symbol: Symbol identifier for the dataset

        Returns:
            Dict with metrics like 'size_bytes', 'object_count', 'last_modified'

        Raises:
            OSError: For filesystem access failures
        """
        dataset_path = self._get_dataset_base_path(symbol)

        if not os.path.exists(dataset_path):
            return {
                'size_bytes': 0,
                'object_count': 0,
                'last_modified': None
            }

        try:
            total_size = 0
            object_count = 0
            last_modified = None

            for root, _, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith('.parquet'):
                        file_path = os.path.join(root, file)
                        file_stat = os.stat(file_path)
                        total_size += file_stat.st_size
                        object_count += 1

                        file_mtime = file_stat.st_mtime
                        if last_modified is None or file_mtime > last_modified:
                            last_modified = file_mtime

            # Convert timestamp to ISO format if available
            if last_modified is not None:
                from datetime import datetime
                last_modified = datetime.fromtimestamp(last_modified).isoformat()

            return {
                'size_bytes': total_size,
                'object_count': object_count,
                'last_modified': last_modified
            }

        except Exception as e:
            logger.error(f"Failed to get storage metrics for {symbol}: {e}")
            raise

    def get_repo_path(self, symbol: str) -> str:
        """
        Get the repository path for storing features for a given symbol.

        For local repository, this returns the symbol-specific directory path
        where partitioned parquet files are stored.

        Args:
            symbol: Symbol identifier for the dataset

        Returns:
            str: The local filesystem path for this symbol's features

        Raises:
            ValueError: If symbol is invalid
            OSError: If directory creation fails
        """
        if not symbol or not symbol.strip():
            raise ValueError("Symbol cannot be empty or None")

        # Construct the symbol-specific directory path
        # Structure: base_path/symbol/year=YYYY/month=MM/day=DD/features_*.parquet
        symbol_path = os.path.join(self.base_path, symbol.strip())

        # Ensure the symbol directory exists
        try:
            os.makedirs(symbol_path, exist_ok=True)
        except OSError as e:
            raise OSError(
                f"Failed to create symbol directory {symbol_path}: {e}"
            ) from e

        return symbol_path
