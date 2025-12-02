"""Local filesystem offline feature repository implementation."""
import os
import logging
import json
from typing import Any, Dict, List, Optional
import pandas as pd
from pandas import DataFrame, concat, to_datetime
from injector import inject
from drl_trading_common.config.feature_config import FeatureStoreConfig
from drl_trading_adapter.adapter.feature_store.offline.parquet.base_parquet_feature_repo import BaseParquetFeatureRepo

logger = logging.getLogger(__name__)


@inject
class OfflineLocalParquetFeatureRepo(BaseParquetFeatureRepo):
    """Stores offline features in a local partitioned parquet layout."""

    def __init__(self, config: FeatureStoreConfig):
        if not config.local_repo_config:
            raise ValueError("local_repo_config required for LOCAL strategy")
        self.config_path = config.config_directory
        self.base_path = config.local_repo_config.repo_path

    def get_repo_path(self, symbol: str) -> str:
        path = os.path.join(self.base_path, symbol)
        return path

    def get_correct_path(self, symbol: str) -> str:
        """Get the correct repository path for a given symbol."""
        path = os.path.join(self.config_path, self.base_path, symbol)
        os.makedirs(path, exist_ok=True)
        return path

    # ==================================================================================
    # Implement abstract I/O methods from BaseOfflineFeatureRepo
    # ==================================================================================

    def _load_metadata(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Load metadata for a symbol (tracks partitions, timestamps, etc.)"""
        metadata_path = os.path.join(self.get_correct_path(symbol), "_metadata.json")
        if not os.path.exists(metadata_path):
            return None
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata for {symbol}: {e}")
            return None

    def _save_metadata(self, symbol: str, metadata: Dict[str, Any]) -> None:
        """Save metadata for a symbol."""
        metadata_path = os.path.join(self.get_correct_path(symbol), "_metadata.json")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata for {symbol}: {e}")

    def _load_partitions(self, symbol: str, partitions: List[Dict]) -> DataFrame:
        """Load data from specific partitions only."""
        dfs = []
        for partition in partitions:
            partition_path = partition["path"]
            try:
                df = pd.read_parquet(partition_path)
                if "event_timestamp" in df.columns:
                    df["event_timestamp"] = to_datetime(df["event_timestamp"])
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load partition {partition_path}: {e}")

        if not dfs:
            return DataFrame()

        return concat(dfs, ignore_index=True).sort_values("event_timestamp")

    def _delete_partitions(self, symbol: str, partitions: List[Dict]) -> None:
        """Delete specific partition files."""
        for partition in partitions:
            partition_path = partition["path"]
            try:
                if os.path.exists(partition_path):
                    os.remove(partition_path)
                    logger.debug(f"Deleted partition: {partition_path}")
            except Exception as e:
                logger.error(f"Failed to delete partition {partition_path}: {e}")

    def _store_with_datetime_organization(self, features_df: DataFrame, symbol: str) -> List[Dict[str, Any]]:
        """Store features in partitioned parquet files and return file metadata."""
        base = self.get_correct_path(symbol)
        features_df = features_df.copy()
        features_df["_date"] = features_df["event_timestamp"].dt.date
        stored_files = []

        for date_key, group_df in features_df.groupby("_date"):
            year, month, day = date_key.year, date_key.month, date_key.day
            target = os.path.join(base, f"year={year}", f"month={month:02d}", f"day={day:02d}")
            os.makedirs(target, exist_ok=True)
            start = group_df["event_timestamp"].min()
            end = group_df["event_timestamp"].max()
            fname = f"features_{start.strftime('%Y%m%d_%H%M%S')}_{end.strftime('%H%M%S')}.parquet"
            file_path = os.path.join(target, fname)
            group_df.drop(columns=["_date"]).to_parquet(file_path, index=False, compression="snappy")

            stored_files.append({
                "path": file_path,
                "min_timestamp": start.isoformat(),
                "max_timestamp": end.isoformat(),
                "record_count": len(group_df)
            })

        return stored_files
