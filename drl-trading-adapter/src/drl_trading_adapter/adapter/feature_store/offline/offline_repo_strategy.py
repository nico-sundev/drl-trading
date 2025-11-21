"""Selects offline repository implementation based on config."""

import logging

from injector import inject

from drl_trading_adapter.adapter.feature_store.offline.offline_feature_repo_interface import (
    IOfflineFeatureRepository,
)
from drl_trading_adapter.adapter.feature_store.offline.parquet.offline_local_parquet_feature_repo import (
    OfflineLocalParquetFeatureRepo,
)
from drl_trading_adapter.adapter.feature_store.offline.parquet.offline_s3_parquet_feature_repo import (
    OfflineS3ParquetFeatureRepo,
)
from drl_trading_common.config.feature_config import FeatureStoreConfig
from drl_trading_common.enum.offline_repo_strategy_enum import OfflineRepoStrategyEnum

# Implementations will be imported lazily inside method to avoid circular issues
logger = logging.getLogger(__name__)


class OfflineRepoStrategy:
    """Factory selecting the offline repo implementation using configured strategy."""

    @inject
    def __init__(self, feature_store_config: FeatureStoreConfig):
        self.config = feature_store_config

    def create_offline_repository(self) -> IOfflineFeatureRepository:
        """Return the concrete offline repository per configuration."""
        strat = self.config.offline_repo_strategy
        if strat == OfflineRepoStrategyEnum.LOCAL:
            return OfflineLocalParquetFeatureRepo(self.config)
        if strat == OfflineRepoStrategyEnum.S3:
            return OfflineS3ParquetFeatureRepo(self.config)
        raise ValueError(f"Unsupported offline repo strategy: {strat}")
