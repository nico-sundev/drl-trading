import logging
from abc import ABC, abstractmethod

from drl_trading_common.config.feature_config import FeatureStoreConfig
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.model.dataset_identifier import DatasetIdentifier
from drl_trading_common.model.feature_config_version_info import (
    FeatureConfigVersionInfo,
)
from injector import inject
from pandas import DataFrame

from drl_trading_core.preprocess.feature_store.offline_store.offline_feature_repo_interface import (
    OfflineFeatureRepoInterface,
)
from drl_trading_core.preprocess.feature_store.provider.feast_provider import (
    FeastProvider,
)

from .feature_view_constants import (
    OBSERVATION_SPACE_FEATURE_VIEW_NAME,
    REWARD_SPACE_FEATURE_VIEW_NAME,
)

logger = logging.getLogger(__name__)


class FeatureStoreSaveRepoInterface(ABC):

    @abstractmethod
    def store_computed_features_offline(
        self,
        features_df: DataFrame,
        dataset_id: DatasetIdentifier,
        feature_version_info: FeatureConfigVersionInfo,
    ) -> None:
        """
        Store computed features in the feature store.

        Args:
            features_df: DataFrame containing the computed features
            dataset_id: Identifier for the dataset to which these features belong
        """
        pass

    @abstractmethod
    def batch_materialize_features(
        self, features_df: DataFrame, dataset_id: DatasetIdentifier
    ) -> None:
        """
        Store computed features in the feature store for online serving.

        Args:
            features_df: DataFrame containing the computed features
            dataset_id: Identifier for the dataset to which these features belong
        """
        pass

    @abstractmethod
    def push_features_to_online_store(
        self, features_df: DataFrame, dataset_id: DatasetIdentifier
    ) -> None:
        """
        Push features directly to online store for real-time inference.

        Bypasses offline storage for single-record inference scenarios
        to avoid file fragmentation and improve performance.

        Args:
            features_df: DataFrame containing the computed features (typically single record)
            dataset_id: Identifier for the dataset to which these features belong
        """
        pass

    @abstractmethod
    def is_enabled(self) -> bool:
        """
        Check if the feature store is enabled.

        :return: True if the feature store is enabled, False otherwise.
        """
        pass


@inject
class FeatureStoreSaveRepo(FeatureStoreSaveRepoInterface):
    """
    Repository for saving features to a feature store.

    This class focuses solely on Feast feature store operations,
    delegating the actual file/S3 storage to OfflineFeatureRepoInterface implementations.
    This follows the Single Responsibility Principle by separating:
    - Feature store orchestration (this class)
    - Storage backend operations (OfflineFeatureRepoInterface implementations)
    """

    def __init__(
        self,
        config: FeatureStoreConfig,
        feast_provider: FeastProvider,
        offline_repo: OfflineFeatureRepoInterface
    ):
        self.config = config
        self.feast_provider = feast_provider
        self.feature_store = feast_provider.get_feature_store()
        self.offline_repo = offline_repo

    def store_computed_features_offline(
        self,
        features_df: DataFrame,
        dataset_id: DatasetIdentifier,
        feature_version_info: FeatureConfigVersionInfo,
    ) -> None:
        """
        Store computed features using the configured offline repository.

        This method:
        1. Delegates storage to the offline repository implementation
        2. Creates and applies Feast feature views
        3. Registers feature services for the dataset

        Args:
            features_df: DataFrame containing the computed features
            dataset_id: Identifier for the dataset
            feature_version_info: Version information for feature tracking
        """
        # Validate input
        if features_df.empty:
            logger.info(f"No features to store for {dataset_id.symbol}/{dataset_id.timeframe.value}")
            return

        if "event_timestamp" not in features_df.columns:
            raise ValueError("features_df must contain 'event_timestamp' column for feature store operations")

        # Store features using the configured offline repository
        stored_count = self.offline_repo.store_features_incrementally(features_df, dataset_id)

        if stored_count == 0:
            logger.info(f"No new features stored for {dataset_id.symbol}/{dataset_id.timeframe.value}")
            return

        logger.info(f"Stored {stored_count} feature records for {dataset_id.symbol}/{dataset_id.timeframe.value}")        # Create and apply Feast feature views
        self._create_and_apply_feature_views(dataset_id, feature_version_info)

    def batch_materialize_features(
        self, features_df: DataFrame, dataset_id: DatasetIdentifier
    ) -> None:
        """
        Materialize features for online serving (batch mode only).

        This method is designed for training/batch processing where large
        datasets are materialized from offline storage to online store.

        For inference mode, use push_features_to_online_store() instead
        to avoid unnecessary offline storage and improve performance.

        Args:
            features_df: DataFrame containing the computed features
            dataset_id: Identifier for the dataset to which these features belong
        """
        if "event_timestamp" not in features_df.columns:
            raise ValueError("features_df must contain 'event_timestamp' column for materialization")

        self.feature_store.materialize(
            start_date=features_df["event_timestamp"].min(),
            end_date=features_df["event_timestamp"].max(),
        )
        logger.info(f"Materialized features for online serving: {dataset_id.symbol}/{dataset_id.timeframe.value}")

    def push_features_to_online_store(
        self, features_df: DataFrame, dataset_id: DatasetIdentifier
    ) -> None:
        """
        Push features directly to online store for real-time inference.

        This method bypasses offline storage and directly updates the online
        feature store, optimized for single-record inference scenarios.

        Use this method for:
        - Real-time inference where features are computed on-demand
        - Single-record updates that don't warrant offline storage
        - High-frequency scenarios where storage I/O would add latency

        Args:
            features_df: DataFrame containing the computed features (typically single record)
            dataset_id: Identifier for the dataset to which these features belong
        """
        if "event_timestamp" not in features_df.columns:
            raise ValueError("features_df must contain 'event_timestamp' column for online store update")

        # Push features directly to online store without offline storage
        # This avoids creating many small parquet files during inference
        # Use Feast's write_to_online_store for direct online updates
        self.feature_store.write_to_online_store(
            feature_view_name=f"{dataset_id.symbol}_{dataset_id.timeframe.value}_features",
            df=features_df
        )
        logger.debug(f"Pushed {len(features_df)} feature records to online store: {dataset_id.symbol}/{dataset_id.timeframe.value}")

    def is_enabled(self) -> bool:
        """
        Check if the feature store is enabled.

        :return: True if the feature store is enabled, False otherwise.
        """
        return self.config.enabled

    def _create_and_apply_feature_views(
        self,
        dataset_id: DatasetIdentifier,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """
        Create and apply Feast feature views and services.

        Args:
            dataset_id: Dataset identifier
            feature_version_info: Version information for feature tracking
        """
        # Create feature views for observation and reward spaces
        obs_fv = self.feast_provider.create_feature_view(
            dataset_id=dataset_id,
            feature_view_name=OBSERVATION_SPACE_FEATURE_VIEW_NAME,
            feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature_version_info=feature_version_info,
        )
        reward_fv = self.feast_provider.create_feature_view(
            dataset_id=dataset_id,
            feature_view_name=REWARD_SPACE_FEATURE_VIEW_NAME,
            feature_role=FeatureRoleEnum.REWARD_ENGINEERING,
            feature_version_info=feature_version_info,
        )

        # Create feature service combining both views
        feature_service = self.feast_provider.create_feature_service(
            feature_views=[obs_fv, reward_fv],
            dataset_id=dataset_id,
            feature_version_info=feature_version_info,
        )

        # Apply to Feast registry
        self.feature_store.apply([obs_fv, reward_fv, feature_service])
        logger.info(f"Applied Feast feature views and service: {feature_service.name}")
