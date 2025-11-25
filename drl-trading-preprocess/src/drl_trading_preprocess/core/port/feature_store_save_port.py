
from abc import ABC, abstractmethod

from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_core.core.dto.offline_storage_request import OfflineStorageRequest
from pandas import DataFrame


class IFeatureStoreSavePort(ABC):

    @abstractmethod
    def store_computed_features_offline(
        self,
        request: OfflineStorageRequest,
    ) -> None:
        """
        Store computed features in the feature store.

        Args:
            request: Offline storage request containing all necessary parameters
        """
        pass

    @abstractmethod
    def batch_materialize_features(
        self,
        features_df: DataFrame,
        symbol: str,
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
        self,
        features_df: DataFrame,
        symbol: str,
        feature_role: FeatureRoleEnum,
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
