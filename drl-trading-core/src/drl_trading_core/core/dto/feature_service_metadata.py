from dataclasses import dataclass
from typing import List

from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.adapter.model.feature_config_version_info import (
    FeatureConfigVersionInfo,
)
from drl_trading_common.core.model.timeframe import Timeframe
from drl_trading_common.core.model.dataset_identifier import DatasetIdentifier
from drl_trading_core.core.dto.feature_view_metadata import FeatureViewMetadata


@dataclass
class FeatureServiceMetadata:

    dataset_identifier: DatasetIdentifier
    feature_service_role: FeatureRoleEnum
    feature_version_info: FeatureConfigVersionInfo
    feature_view_metadata_list: List[FeatureViewMetadata]

    def validate(self) -> None:
        """
        Validate all parameters in the request.

        Raises:
            ValueError: If any parameter is invalid or missing required attributes
        """
        # Dataset identifier validation
        if not isinstance(self.dataset_identifier, DatasetIdentifier):
            raise ValueError(f"Dataset identifier must be a DatasetIdentifier, got {type(self.dataset_identifier)}")

        # Validate symbol within dataset_identifier
        if (
            not self.dataset_identifier.symbol
            or not isinstance(self.dataset_identifier.symbol, str)
            or not self.dataset_identifier.symbol.strip()
        ):
            raise ValueError("Symbol must be a non-empty string")

        # Validate timeframe within dataset_identifier
        if not isinstance(self.dataset_identifier.timeframe, Timeframe):
            raise ValueError(f"Timeframe must be a Timeframe, got {type(self.dataset_identifier.timeframe)}")

        # Feature role validation
        if not isinstance(self.feature_service_role, FeatureRoleEnum):
            raise ValueError(
                f"Feature role must be a FeatureRoleEnum, got {type(self.feature_service_role)}"
            )

        # Feature validation
        if not isinstance(self.feature_version_info, FeatureConfigVersionInfo):
            raise ValueError(
                f"Feature must be a FeatureConfigVersionInfo, got {type(self.feature_version_info)}"
            )

        # Validate feature definitions exist
        if not self.feature_version_info.feature_definitions:
            raise ValueError("Feature definitions must not be empty")

        # Feature view metadata list validation
        if not isinstance(self.feature_view_metadata_list, list):
            raise ValueError("Feature view metadata list must be a list")

        for metadata in self.feature_view_metadata_list:
            if not isinstance(metadata, FeatureViewMetadata):
                raise ValueError("All items in feature_view_metadata_list must be FeatureViewMetadata instances")

    def get_sanitized_symbol(self) -> str:
        """Get sanitized symbol string."""
        return self.dataset_identifier.symbol.strip() if self.dataset_identifier.symbol else ""

    @classmethod
    def create(
        cls,
        dataset_identifier: DatasetIdentifier,
        feature_role: FeatureRoleEnum,
        feature_config_version: FeatureConfigVersionInfo,
        feature_view_metadata_list: List[FeatureViewMetadata],
    ) -> "FeatureServiceMetadata":
        """
        Factory method to create and validate a FeatureServiceRequestContainer.

        Args:
            dataset_identifier: The dataset identifier
            feature_role: The role of the feature in the view
            feature_config_version: The feature configuration version info
            feature_view_metadata_list: List of feature view metadata

        Returns:
            FeatureServiceRequestContainer: Validated request object

        Raises:
            ValueError: If any parameter is invalid
        """
        request = cls(
            dataset_identifier=dataset_identifier,
            feature_service_role=feature_role,
            feature_version_info=feature_config_version,
            feature_view_metadata_list=feature_view_metadata_list,
        )
        request.validate()
        return request
