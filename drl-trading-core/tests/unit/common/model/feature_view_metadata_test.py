"""
Test for the FeatureViewRequest dataclass.

This demonstrates the container pattern for feature view creation parameters.
"""
from typing import Any

import pytest
from drl_trading_common.core.model.feature_metadata import FeatureMetadata
from drl_trading_common.core.model.dataset_identifier import DatasetIdentifier
from drl_trading_common.core.model.timeframe import Timeframe

from drl_trading_core.core.dto.feature_view_metadata import FeatureViewMetadata


class TestFeatureViewRequest:
    """Test the FeatureViewRequest dataclass."""

    def test_create_valid_request(self, mock_feature_metadata: FeatureMetadata, mock_dataset_identifier: DatasetIdentifier) -> None:
        """Test creating a valid feature view request."""
        # Given

        # When
        request = FeatureViewMetadata.create(
            dataset_identifier=mock_dataset_identifier,
            feature_metadata=mock_feature_metadata
        )

        # Then
        assert request.dataset_identifier == mock_dataset_identifier
        assert request.feature_metadata == mock_feature_metadata

    def test_direct_instantiation(self, mock_feature_metadata: FeatureMetadata, mock_dataset_identifier: DatasetIdentifier) -> None:
        """Test direct instantiation of FeatureViewRequest."""
        # Given

        # When
        request = FeatureViewMetadata(
            dataset_identifier=mock_dataset_identifier,
            feature_metadata=mock_feature_metadata
        )

        # Then
        assert request.dataset_identifier == mock_dataset_identifier
        assert request.feature_metadata == mock_feature_metadata

    def test_validation_empty_symbol(self, mock_feature_metadata: FeatureMetadata, mock_timeframe: Timeframe) -> None:
        """Test validation fails for empty symbol."""
        # Given
        empty_dataset_identifier = DatasetIdentifier(symbol="", timeframe=mock_timeframe)

        # When/Then
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            FeatureViewMetadata.create(
                dataset_identifier=empty_dataset_identifier,
                feature_metadata=mock_feature_metadata
            )

    def test_validation_whitespace_only_symbol(self, mock_feature_metadata: FeatureMetadata, mock_timeframe: Timeframe) -> None:
        """Test validation fails for whitespace-only symbol."""
        # Given
        whitespace_dataset_identifier = DatasetIdentifier(symbol="   ", timeframe=mock_timeframe)

        # When/Then
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            FeatureViewMetadata.create(
                dataset_identifier=whitespace_dataset_identifier,
                feature_metadata=mock_feature_metadata
            )

    def test_validation_invalid_dataset_identifier(self, mock_feature_metadata: FeatureMetadata) -> None:
        """Test validation fails for invalid dataset identifier."""
        # Given/When/Then
        with pytest.raises(ValueError, match="Dataset identifier must be a DatasetIdentifier"):
            FeatureViewMetadata.create(
                dataset_identifier="not_a_dataset_identifier",  # type: ignore[arg-type]
                feature_metadata=mock_feature_metadata
            )

    @pytest.mark.parametrize("invalid_feature_metadata", [
        "not_a_feature_metadata",
        123,
        [],
        {},
        None
    ])
    def test_validation_invalid_feature_metadata(self, invalid_feature_metadata: Any, mock_dataset_identifier: DatasetIdentifier) -> None:
        """Test validation fails for invalid feature metadata types."""
        # Given

        # When/Then
        with pytest.raises(ValueError, match="Feature metadata must be a FeatureMetadata"):
            FeatureViewMetadata.create(
                dataset_identifier=mock_dataset_identifier,
                feature_metadata=invalid_feature_metadata  # type: ignore[arg-type]
            )

    def test_get_sanitized_symbol(self, mock_feature_metadata: FeatureMetadata, mock_timeframe: Timeframe) -> None:
        """Test sanitized symbol accessor method."""
        # Given
        symbol_with_whitespace = "  EURUSD  "
        dataset_identifier = DatasetIdentifier(symbol=symbol_with_whitespace, timeframe=mock_timeframe)
        request = FeatureViewMetadata(
            dataset_identifier=dataset_identifier,
            feature_metadata=mock_feature_metadata
        )

        # When
        sanitized_symbol = request.get_sanitized_symbol()

        # Then
        assert sanitized_symbol == "EURUSD"

    def test_get_sanitized_symbol_empty_string(self, mock_feature_metadata: FeatureMetadata, mock_timeframe: Timeframe) -> None:
        """Test sanitized symbol returns empty string for empty symbol."""
        # Given
        dataset_identifier = DatasetIdentifier(symbol="", timeframe=mock_timeframe)
        request = FeatureViewMetadata(
            dataset_identifier=dataset_identifier,
            feature_metadata=mock_feature_metadata
        )

        # When
        sanitized_symbol = request.get_sanitized_symbol()

        # Then
        assert sanitized_symbol == ""

    def test_validation_called_on_create(self, mock_feature_metadata: FeatureMetadata, mock_dataset_identifier: DatasetIdentifier) -> None:
        """Test that validation is called when using create factory method."""
        # Given

        # When
        request = FeatureViewMetadata.create(
            dataset_identifier=mock_dataset_identifier,
            feature_metadata=mock_feature_metadata
        )

        # Then
        # Should not raise any exception
        assert request is not None
        assert request.dataset_identifier == mock_dataset_identifier
        assert request.feature_metadata == mock_feature_metadata

    def test_manual_validation_call(self, mock_feature_metadata: FeatureMetadata, mock_dataset_identifier: DatasetIdentifier) -> None:
        """Test calling validation manually on a valid request."""
        # Given
        request = FeatureViewMetadata(
            dataset_identifier=mock_dataset_identifier,
            feature_metadata=mock_feature_metadata
        )

        # When/Then
        # Should not raise any exception
        request.validate()


class TestFeatureViewRequestBenefits:
    """Demonstrate the benefits of the container pattern."""

    def test_readability_improvement(self, mock_feature_metadata: FeatureMetadata, mock_dataset_identifier: DatasetIdentifier) -> None:
        """
        Demonstrate how the container improves readability.

        Compare:
        OLD: provider.create_feature_view("EURUSD", FeatureRoleEnum.OBSERVATION_SPACE, feature, timeframe)
        NEW: provider.create_feature_view_from_request(request)
        """
        # Given
        # New approach: Self-documenting and clear
        request = FeatureViewMetadata.create(
            dataset_identifier=mock_dataset_identifier,
            feature_metadata=mock_feature_metadata
        )

        # Then
        # The container makes the intent clear and groups related parameters
        assert request.dataset_identifier == mock_dataset_identifier
        assert request.feature_metadata == mock_feature_metadata
        # Parameters are validated as a unit
        # Better testability - can create test fixtures easily

    def test_extensibility_benefit(self, mock_feature_metadata: FeatureMetadata, mock_dataset_identifier: DatasetIdentifier) -> None:
        """
        Demonstrate how the container makes the API more extensible.

        Adding new parameters to the container doesn't break existing code.
        """
        # Given - we could easily add new fields to FeatureViewRequest:
        # - description: Optional[str] = None
        # - tags: Dict[str, str] = field(default_factory=dict)
        # - priority: int = 0
        #
        # Without breaking any existing calls!

        request = FeatureViewMetadata.create(
            dataset_identifier=mock_dataset_identifier,
            feature_metadata=mock_feature_metadata
        )

        # When/Then
        # This pattern allows for future enhancement without API breaks
        assert request is not None
