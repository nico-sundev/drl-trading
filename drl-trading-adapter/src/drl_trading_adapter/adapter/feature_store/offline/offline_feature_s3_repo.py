"""S3 offline feature repository (minimal placeholder)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pandas import DataFrame
from injector import inject
from drl_trading_common.config.feature_config import FeatureStoreConfig
from drl_trading_adapter.adapter.feature_store.offline.offline_feature_repo_interface import IOfflineFeatureRepository


@inject
class OfflineFeatureS3Repo(IOfflineFeatureRepository):  # pragma: no cover - placeholder
    """Placeholder S3 implementation (to be fully implemented when S3 integration added)."""

    def __init__(self, config: FeatureStoreConfig):  # noqa: D401
        self.config = config

    def get_repo_path(self, symbol: str) -> str:  # S3 URI placeholder
        return f"s3://{self.config.s3_repo_config.bucket_name}/{symbol}" if self.config.s3_repo_config else symbol

    def store_features_incrementally(self, features_df: DataFrame, symbol: str) -> int:
        return 0

    def load_existing_features(self, symbol: str) -> Optional[DataFrame]:
        return None

    def feature_exists(self, symbol: str) -> bool:
        return False

    def get_feature_count(self, symbol: str) -> int:
        return 0

    def store_features_batch(self, feature_batches: List[Dict[str, Any]]) -> Dict[str, int]:
        return {b["symbol"]: 0 for b in feature_batches}

    def delete_features(self, symbol: str) -> bool:
        return False

    def get_storage_metrics(self, symbol: str) -> Dict[str, Any]:
        return {"size_bytes": 0, "object_count": 0, "last_modified": None}
