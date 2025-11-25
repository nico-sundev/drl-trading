from dataclasses import dataclass

from drl_trading_common.core.model.feature_metadata import FeatureMetadata
from drl_trading_common.core.model.timeframe import Timeframe
from drl_trading_common.core.model.dataset_identifier import DatasetIdentifier


@dataclass
class FeatureViewMetadata:
    """
    Container for feature view creation parameters.

    This encapsulates all the information needed to create a Feast feature view,
    improving readability and maintainability by grouping related parameters.

    Attributes:
        dataset_identifier: The dataset identifier for the feature view
        feature_metadata: The feature metadata to include in the view
    """

    dataset_identifier: DatasetIdentifier
    feature_metadata: FeatureMetadata

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

        # Feature metadata validation
        if not isinstance(self.feature_metadata, FeatureMetadata):
            raise ValueError(f"Feature metadata must be a FeatureMetadata, got {type(self.feature_metadata)}")

    def get_sanitized_symbol(self) -> str:
        """Get sanitized symbol string."""
        return self.dataset_identifier.symbol.strip() if self.dataset_identifier.symbol else ""

    @classmethod
    def create(
        cls, dataset_identifier: DatasetIdentifier, feature_metadata: FeatureMetadata
    ) -> "FeatureViewMetadata":
        """
        Factory method to create and validate a FeatureViewRequest.

        Args:
            dataset_identifier: The dataset identifier for the feature view
            feature_metadata: The feature metadata to include in the view

        Returns:
            FeatureViewRequest: Validated request object

        Raises:
            ValueError: If any parameter is invalid
        """
        request = cls(dataset_identifier=dataset_identifier, feature_metadata=feature_metadata)
        request.validate()
        return request
