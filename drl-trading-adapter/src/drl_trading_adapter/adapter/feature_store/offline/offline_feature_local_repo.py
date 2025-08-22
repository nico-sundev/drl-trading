"""Local filesystem offline feature repository implementation."""
import os
import logging
from typing import Any, Dict, List, Optional
import pandas as pd
from pandas import DataFrame, concat, to_datetime
from injector import inject
from drl_trading_common.config.feature_config import FeatureStoreConfig
from drl_trading_core.preprocess.feature_store.port.offline_feature_repo_interface import IOfflineFeatureRepository

logger = logging.getLogger(__name__)


@inject
class OfflineFeatureLocalRepo(IOfflineFeatureRepository):
    """Stores offline features in a local partitioned parquet layout."""

    def __init__(self, config: FeatureStoreConfig):
        if not config.local_repo_config:
            raise ValueError("local_repo_config required for LOCAL strategy")
        self.base_path = config.local_repo_config.repo_path

    def get_repo_path(self, symbol: str) -> str:
        path = os.path.join(self.base_path, symbol)
        os.makedirs(path, exist_ok=True)
        return path

    def store_features_incrementally(self, features_df: DataFrame, symbol: str) -> int:
        if features_df.empty:
            return 0
        if "event_timestamp" not in features_df.columns:
            raise ValueError("'event_timestamp' column required")
        features_df = features_df.copy()
        features_df["event_timestamp"] = to_datetime(features_df["event_timestamp"])
        features_df = features_df.sort_values("event_timestamp").drop_duplicates(
            subset=["event_timestamp"]
        )
        existing = self.load_existing_features(symbol)
        if existing is not None and not existing.empty:
            existing_ts = set(existing["event_timestamp"])
            features_df = features_df[~features_df["event_timestamp"].isin(existing_ts)]
        if features_df.empty:
            return 0
        self._store_partitioned(features_df, symbol)
        return len(features_df)

    def _store_partitioned(self, features_df: DataFrame, symbol: str) -> None:
        base = self.get_repo_path(symbol)
        features_df = features_df.copy()
        features_df["_date"] = features_df["event_timestamp"].dt.date
        for date_key, group_df in features_df.groupby("_date"):
            year, month, day = date_key.year, date_key.month, date_key.day
            target = os.path.join(base, f"year={year}", f"month={month:02d}", f"day={day:02d}")
            os.makedirs(target, exist_ok=True)
            start = group_df["event_timestamp"].min()
            end = group_df["event_timestamp"].max()
            fname = f"features_{start.strftime('%Y%m%d_%H%M%S')}_{end.strftime('%H%M%S')}.parquet"
            group_df.drop(columns=["_date"]).to_parquet(os.path.join(target, fname), index=False, compression="snappy")

    def load_existing_features(self, symbol: str) -> Optional[DataFrame]:
        base = self.get_repo_path(symbol)
        parquet_files: list[str] = []
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, f))
        if not parquet_files:
            return None
        dfs = []
        for fp in parquet_files:
            try:
                df = pd.read_parquet(fp)
                if "event_timestamp" in df.columns:
                    df["event_timestamp"] = to_datetime(df["event_timestamp"])
                dfs.append(df)
            except Exception as e:  # pragma: no cover
                logger.warning("Failed loading %s: %s", fp, e)
        if not dfs:
            return None
        return (
            concat(dfs, ignore_index=True)
            .drop_duplicates(subset=["event_timestamp"])
            .sort_values("event_timestamp")
        )

    def feature_exists(self, symbol: str) -> bool:
        return self.load_existing_features(symbol) is not None

    def get_feature_count(self, symbol: str) -> int:
        df = self.load_existing_features(symbol)
        return 0 if df is None else len(df)

    def store_features_batch(self, feature_batches: List[Dict[str, Any]]) -> Dict[str, int]:  # pragma: no cover
        return {b["symbol"]: self.store_features_incrementally(b["features_df"], b["symbol"]) for b in feature_batches}

    def delete_features(self, symbol: str) -> bool:  # pragma: no cover
        import shutil
        path = self.get_repo_path(symbol)
        if not os.path.exists(path):
            return False
        shutil.rmtree(path)
        return True

    def get_storage_metrics(self, symbol: str) -> Dict[str, Any]:  # pragma: no cover
        return {"size_bytes": 0, "object_count": 0, "last_modified": None}
