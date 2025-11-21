"""
Test for the FeatureViewRequest dataclass.

This demonstrates the container pattern for feature view creation parameters.
"""
from typing import Any

import pytest
from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.model.timeframe import Timeframe

from drl_trading_core.common.model.feature_view_metadata import FeatureViewMetadata


class TestFeatureViewRequest:
    """Test the FeatureViewRequest dataclass."""

    def test_create_valid_request(self, mock_feature: BaseFeature, mock_timeframe: Timeframe) -> None:
        """Test creating a valid feature view request."""
        # Given
        symbol = "EURUSD"
        feature_role = FeatureRoleEnum.OBSERVATION_SPACE

        # When
        request = FeatureViewMetadata.create(
            symbol=symbol,
            feature_role=feature_role,
            feature=mock_feature,
            timeframe=mock_timeframe
        )

        # Then
        assert request.symbol == "EURUSD"
        assert request.feature_role == FeatureRoleEnum.OBSERVATION_SPACE
        assert request.feature == mock_feature
        assert request.timeframe == mock_timeframe

    def test_direct_instantiation(self, mock_feature: BaseFeature, mock_timeframe: Timeframe) -> None:
        """Test direct instantiation of FeatureViewRequest."""
        # Given
        symbol = "EURUSD"
        feature_role = FeatureRoleEnum.OBSERVATION_SPACE

        # When
        request = FeatureViewMetadata(
            symbol=symbol,
            feature_role=feature_role,
            feature=mock_feature,
            timeframe=mock_timeframe
        )

        # Then
        assert request.symbol == "EURUSD"
        assert request.feature_role == FeatureRoleEnum.OBSERVATION_SPACE
        assert request.feature == mock_feature
        assert request.timeframe == mock_timeframe

    def test_validation_empty_symbol(self, mock_feature: BaseFeature, mock_timeframe: Timeframe) -> None:
        """Test validation fails for empty symbol."""
        # Given
        empty_symbol = ""

        # When/Then
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            FeatureViewMetadata.create(
                symbol=empty_symbol,
                feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
                feature=mock_feature,
                timeframe=mock_timeframe
            )

    def test_validation_whitespace_only_symbol(self, mock_feature: BaseFeature, mock_timeframe: Timeframe) -> None:
        """Test validation fails for whitespace-only symbol."""
        # Given
        whitespace_symbol = "   "

        # When/Then
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            FeatureViewMetadata.create(
                symbol=whitespace_symbol,
                feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
                feature=mock_feature,
                timeframe=mock_timeframe
            )

    def test_validation_none_symbol(self, mock_feature: BaseFeature, mock_timeframe: Timeframe) -> None:
        """Test validation fails for None symbol."""
        # Given/When/Then
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            FeatureViewMetadata.create(
                symbol=None,  # type: ignore[arg-type]
                feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
                feature=mock_feature,
                timeframe=mock_timeframe
            )

    @pytest.mark.parametrize("invalid_role", [
        "not_an_enum",
        123,
        [],
        {},
        None
    ])
    def test_validation_invalid_feature_role(self, invalid_role: Any, mock_feature: BaseFeature, mock_timeframe: Timeframe) -> None:
        """Test validation fails for invalid feature role types."""
        # Given
        symbol = "EURUSD"

        # When/Then
        with pytest.raises(ValueError, match="Feature role must be a FeatureRoleEnum"):
            FeatureViewMetadata.create(
                symbol=symbol,
                feature_role=invalid_role,  # type: ignore[arg-type]
                feature=mock_feature,
                timeframe=mock_timeframe
            )

    @pytest.mark.parametrize("invalid_feature", [
        "not_a_feature",
        123,
        [],
        {},
        None
    ])
    def test_validation_invalid_feature(self, invalid_feature: Any, mock_timeframe: Timeframe) -> None:
        """Test validation fails for invalid feature types."""
        # Given
        symbol = "EURUSD"
        feature_role = FeatureRoleEnum.OBSERVATION_SPACE

        # When/Then
        with pytest.raises(ValueError, match="Feature must be a BaseFeature"):
            FeatureViewMetadata.create(
                symbol=symbol,
                feature_role=feature_role,
                feature=invalid_feature,  # type: ignore[arg-type]
                timeframe=mock_timeframe
            )

    @pytest.mark.parametrize("invalid_timeframe", [
        "not_a_timeframe",
        123,
        [],
        {},
        None
    ])
    def test_validation_invalid_timeframe(self, invalid_timeframe: Any, mock_feature: BaseFeature) -> None:
        """Test validation fails for invalid timeframe types."""
        # Given
        symbol = "EURUSD"
        feature_role = FeatureRoleEnum.OBSERVATION_SPACE

        # When/Then
        with pytest.raises(ValueError, match="Timeframe must be a Timeframe"):
            FeatureViewMetadata.create(
                symbol=symbol,
                feature_role=feature_role,
                feature=mock_feature,
                timeframe=invalid_timeframe  # type: ignore[arg-type]
            )

    def test_get_sanitized_symbol(self, mock_feature: BaseFeature, mock_timeframe: Timeframe) -> None:
        """Test sanitized symbol accessor method."""
        # Given
        symbol_with_whitespace = "  EURUSD  "
        request = FeatureViewMetadata(
            symbol=symbol_with_whitespace,
            feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature=mock_feature,
            timeframe=mock_timeframe
        )

        # When
        sanitized_symbol = request.get_sanitized_symbol()

        # Then
        assert sanitized_symbol == "EURUSD"

    def test_get_sanitized_symbol_empty_string(self, mock_feature: BaseFeature, mock_timeframe: Timeframe) -> None:
        """Test sanitized symbol returns empty string for None symbol."""
        # Given
        request = FeatureViewMetadata(
            symbol=None,  # type: ignore[arg-type]
            feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature=mock_feature,
            timeframe=mock_timeframe
        )

        # When
        sanitized_symbol = request.get_sanitized_symbol()

        # Then
        assert sanitized_symbol == ""

    def test_validation_called_on_create(self, mock_feature: BaseFeature, mock_timeframe: Timeframe) -> None:
        """Test that validation is called when using create factory method."""
        # Given
        valid_symbol = "EURUSD"
        valid_role = FeatureRoleEnum.OBSERVATION_SPACE

        # When
        request = FeatureViewMetadata.create(
            symbol=valid_symbol,
            feature_role=valid_role,
            feature=mock_feature,
            timeframe=mock_timeframe
        )

        # Then
        # Should not raise any exception
        assert request is not None
        assert request.symbol == valid_symbol
        assert request.feature_role == valid_role
        assert request.feature == mock_feature
        assert request.timeframe == mock_timeframe

    def test_manual_validation_call(self, mock_feature: BaseFeature, mock_timeframe: Timeframe) -> None:
        """Test calling validation manually on a valid request."""
        # Given
        request = FeatureViewMetadata(
            symbol="EURUSD",
            feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature=mock_feature,
            timeframe=mock_timeframe
        )

        # When/Then
        # Should not raise any exception
        request.validate()


class TestFeatureViewRequestBenefits:
    """Demonstrate the benefits of the container pattern."""

    def test_readability_improvement(self, mock_feature: BaseFeature, mock_timeframe: Timeframe) -> None:
        """
        Demonstrate how the container improves readability.

        Compare:
        OLD: provider.create_feature_view("EURUSD", FeatureRoleEnum.OBSERVATION_SPACE, feature, timeframe)
        NEW: provider.create_feature_view_from_request(request)
        """
        # Given
        # New approach: Self-documenting and clear
        request = FeatureViewMetadata.create(
            symbol="EURUSD",
            feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature=mock_feature,
            timeframe=mock_timeframe
        )

        # Then
        # The container makes the intent clear and groups related parameters
        assert request.symbol == "EURUSD"
        assert request.feature_role == FeatureRoleEnum.OBSERVATION_SPACE
        assert request.feature == mock_feature
        assert request.timeframe == mock_timeframe
        # Parameters are validated as a unit
        # Better testability - can create test fixtures easily

    def test_extensibility_benefit(self, mock_feature: BaseFeature, mock_timeframe: Timeframe) -> None:
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
            symbol="EURUSD",
            feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature=mock_feature,
            timeframe=mock_timeframe
        )

        # When/Then
        # This pattern allows for future enhancement without API breaks
        assert request is not None
