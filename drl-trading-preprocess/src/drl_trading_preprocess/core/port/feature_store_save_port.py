
from abc import ABC, abstractmethod

import pandas as pd
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo
from drl_trading_core.common.model.feature_view_metadata import FeatureViewMetadata
from pandas import DataFrame


class IFeatureStoreSavePort(ABC):

    @abstractmethod
    def store_computed_features_offline(
        self,
        features_df: DataFrame,
        symbol: str,
        feature_version_info: FeatureConfigVersionInfo,
        feature_view_requests: list[FeatureViewMetadata],
        processing_context: str = "training",
        requested_start_time: pd.Timestamp | None = None,
        requested_end_time: pd.Timestamp | None = None,
    ) -> None:
        """
        Store computed features in the feature store.

        Args:
            features_df: DataFrame containing the computed features
            symbol: Symbol identifier
            feature_version_info: Version information for feature tracking
            feature_view_requests: List of feature view requests
            processing_context: Context (backfill/training/inference) determines storage strategy
            requested_start_time: Original requested start time (for filtering backfill results)
            requested_end_time: Original requested end time (for filtering backfill results)
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
