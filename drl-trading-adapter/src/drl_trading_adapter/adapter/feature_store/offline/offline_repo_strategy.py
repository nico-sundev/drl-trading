"""Selects offline repository implementation based on config."""
import logging
from injector import inject
from drl_trading_common.config.feature_config import FeatureStoreConfig
from drl_trading_common.enum.offline_repo_strategy_enum import OfflineRepoStrategyEnum
from drl_trading_core.preprocess.feature_store.offline_store.offline_feature_repo_interface import IOfflineFeatureRepository

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
            from .offline_feature_local_repo import OfflineFeatureLocalRepo  # local import
            return OfflineFeatureLocalRepo(self.config)
        if strat == OfflineRepoStrategyEnum.S3:
            from .offline_feature_s3_repo import OfflineFeatureS3Repo  # local import
            return OfflineFeatureS3Repo(self.config)
        raise ValueError(f"Unsupported offline repo strategy: {strat}")
