from dataclasses import dataclass

import pandas as pd
from drl_trading_core.core.dto.feature_service_metadata import FeatureServiceMetadata
from pandas import DataFrame


@dataclass
class OfflineStorageRequest:
    """
    Request object for offline feature storage.

    Encapsulates all parameters needed to store computed features offline,
    replacing the growing parameter list of store_computed_features_offline.
    """

    features_df: DataFrame
    feature_service_metadata: FeatureServiceMetadata
    processing_context: str = "training"
    requested_start_time: pd.Timestamp | None = None
    requested_end_time: pd.Timestamp | None = None

    @property
    def symbol(self) -> str:
        """Get the symbol from the feature service metadata."""
        return self.feature_service_metadata.dataset_identifier.symbol

    @property
    def feature_version_info(self):
        """Get the feature version info from the feature service metadata."""
        return self.feature_service_metadata.feature_version_info

    @property
    def feature_view_metadata_list(self):
        """Get the feature view metadata list from the feature service metadata."""
        return self.feature_service_metadata.feature_view_metadata_list

    def validate(self) -> None:
        """
        Validate the offline storage request.

        Raises:
            ValueError: If any parameter is invalid
        """
        if not isinstance(self.features_df, DataFrame) or self.features_df.empty:
            raise ValueError("features_df must be a non-empty DataFrame")

        if not isinstance(self.feature_service_metadata, FeatureServiceMetadata):
            raise ValueError("feature_service_metadata must be a FeatureServiceMetadata instance")

        if self.processing_context not in ["training", "inference", "backfill"]:
            raise ValueError("processing_context must be one of: training, inference, backfill")

    @classmethod
    def create(
        cls,
        features_df: DataFrame,
        feature_service_metadata: FeatureServiceMetadata,
        processing_context: str = "training",
        requested_start_time: pd.Timestamp | None = None,
        requested_end_time: pd.Timestamp | None = None,
    ) -> "OfflineStorageRequest":
        """
        Factory method to create and validate an OfflineStorageRequest.

        Args:
            features_df: DataFrame containing computed features
            feature_service_metadata: Feature service metadata containing symbol, version info, and view metadata
            processing_context: Processing context (training/inference/backfill)
            requested_start_time: Original requested start time
            requested_end_time: Original requested end time

        Returns:
            OfflineStorageRequest: Validated request object

        Raises:
            ValueError: If validation fails
        """
        request = cls(
            features_df=features_df,
            feature_service_metadata=feature_service_metadata,
            processing_context=processing_context,
            requested_start_time=requested_start_time,
            requested_end_time=requested_end_time,
        )
        request.validate()
        return request
