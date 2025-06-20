import logging
from abc import ABC, abstractmethod

from drl_trading_common.config.feature_config import FeatureStoreConfig
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.model.dataset_identifier import DatasetIdentifier
from injector import inject
from pandas import DataFrame

from drl_trading_core.preprocess.feast.feast_provider import FeastProvider

from .feature_view_constants import (
    OBSERVATION_SPACE_FEATURE_VIEW_NAME,
    REWARD_SPACE_FEATURE_VIEW_NAME,
)

logger = logging.getLogger(__name__)


class FeatureStoreSaveRepoInterface(ABC):

    @abstractmethod
    def store_computed_features_offline(
        self, features_df: DataFrame, dataset_id: DatasetIdentifier
    ) -> None:
        """
        Store computed features in the feature store.

        Args:
            features_df: DataFrame containing the computed features
            dataset_id: Identifier for the dataset to which these features belong
        """
        pass

    @abstractmethod
    def store_computed_features_online(
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
    """

    def __init__(self, config: FeatureStoreConfig, feast_provider: FeastProvider):
        self.config = config
        self.feast_provider = feast_provider
        self.feature_store = feast_provider.get_feature_store()

    def store_computed_features_offline(
        self, features_df: DataFrame, dataset_id: DatasetIdentifier
    ) -> None:
        file_path = f"{self.config.offline_store_path}/{dataset_id.symbol}/{dataset_id.timeframe.value}/computed_features.parquet"

        # Save features to parquet file
        features_df.to_parquet(
            file_path,
            index=False,
        )

        logger.info(f"Storing computed features in feature store at {file_path}")
        entity = self.feast_provider.get_entity(dataset_id)
        obs_fv = self.feast_provider.create_feature_view(
            entity=entity,
            dataset_id=dataset_id,
            feature_view_name=OBSERVATION_SPACE_FEATURE_VIEW_NAME,
            feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
        )
        reward_fv = self.feast_provider.create_feature_view(
            entity=entity,
            dataset_id=dataset_id,
            feature_view_name=REWARD_SPACE_FEATURE_VIEW_NAME,
            feature_role=FeatureRoleEnum.REWARD_ENGINEERING,
        )
        self.feature_store.apply([entity, obs_fv, reward_fv])

    def store_computed_features_online(
        self, features_df: DataFrame, dataset_id: DatasetIdentifier
    ) -> None:
        """
        Store computed features in the feature store for online serving.

        Args:
            features_df: DataFrame containing the computed features
            dataset_id: Identifier for the dataset to which these features belong
        """
        self.feature_store.materialize(
            start_date=features_df["event_timestamp"].min(),
            end_date=features_df["event_timestamp"].max(),
        )
