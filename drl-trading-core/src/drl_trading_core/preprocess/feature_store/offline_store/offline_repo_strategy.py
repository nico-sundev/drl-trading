"""
Strategy pattern implementation for offline feature repository selection.

This module provides a factory that returns the appropriate offline repository
implementation based on the configuration strategy.
"""

import logging

from drl_trading_common.config.feature_config import FeatureStoreConfig
from drl_trading_common.enum.offline_repo_strategy_enum import OfflineRepoStrategyEnum
from injector import inject

from drl_trading_core.preprocess.feature_store.offline_store.offline_feature_repo_interface import (
    IOfflineFeatureRepository,
)

logger = logging.getLogger(__name__)


class OfflineRepoStrategy:
    """
    Strategy pattern implementation for offline feature repository selection.

    This class determines which offline repository implementation to use
    based on the configuration strategy and creates the appropriate instance.
    """

    @inject
    def __init__(self, feature_store_config: FeatureStoreConfig):
        """
        Initialize the strategy with feature store configuration.

        Args:
            feature_store_config: Configuration containing repository strategy and settings
        """
        self.feature_store_config = feature_store_config

    def create_offline_repository(self) -> IOfflineFeatureRepository:
        """
        Create and return the appropriate offline repository implementation.

        Returns:
            Configured offline repository instance

        Raises:
            ValueError: If strategy is not supported or configuration is invalid
        """
        strategy = self.feature_store_config.offline_repo_strategy

        if strategy == OfflineRepoStrategyEnum.LOCAL:
            return self._create_local_repository()
        elif strategy == OfflineRepoStrategyEnum.S3:
            return self._create_s3_repository()
        else:
            raise ValueError(f"Unsupported offline repository strategy: {strategy}")

    def _create_local_repository(self) -> IOfflineFeatureRepository:
        """Create local filesystem repository with appropriate configuration."""
        from drl_trading_core.preprocess.feature_store.offline_store.offline_feature_local_repo import (
            OfflineFeatureLocalRepo,
        )

        # Use new configuration structure
        if not self.feature_store_config.local_repo_config:
            raise ValueError(
                "Local repository configuration missing. "
                "'local_repo_config' must be provided."
            )

        local_config = self.feature_store_config.local_repo_config

        # Create a temporary config object with the repo_path for backward compatibility
        # This will be removed once the repository classes are updated to use the new configuration structure
        from types import SimpleNamespace
        temp_config = SimpleNamespace()
        temp_config.repo_path = local_config.repo_path

        logger.info(f"Creating local offline repository with path: {temp_config.repo_path}")
        return OfflineFeatureLocalRepo(temp_config)

    def _create_s3_repository(self) -> IOfflineFeatureRepository:
        """Create S3 repository with appropriate configuration."""
        from drl_trading_core.preprocess.feature_store.offline_store.offline_feature_s3_repo import (
            OfflineFeatureS3Repo,
        )

        # Use new configuration structure
        if not self.feature_store_config.s3_repo_config:
            raise ValueError(
                "S3 repository configuration missing. "
                "'s3_repo_config' must be provided."
            )

        s3_config = self.feature_store_config.s3_repo_config

        # Create a temporary config object with the S3 attributes for backward compatibility
        # This will be removed once the repository classes are updated to use the new configuration structure
        from types import SimpleNamespace
        temp_config = SimpleNamespace()
        temp_config.s3_bucket_name = s3_config.bucket_name
        temp_config.s3_prefix = s3_config.prefix
        temp_config.s3_endpoint_url = s3_config.endpoint_url
        temp_config.s3_region = s3_config.region
        temp_config.s3_access_key_id = s3_config.access_key_id
        temp_config.s3_secret_access_key = s3_config.secret_access_key

        logger.info(f"Creating S3 offline repository with bucket: {temp_config.s3_bucket_name}")
        return OfflineFeatureS3Repo(temp_config)
