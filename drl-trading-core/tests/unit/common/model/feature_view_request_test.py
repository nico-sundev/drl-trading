"""
Test for the improved FeatureViewRequest container pattern.

This demonstrates the benefits of using a container class over multiple parameters.
"""
from datetime import datetime
from typing import Any

import pytest
from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.model.feature_config_version_info import (
    FeatureConfigVersionInfo,
)

from drl_trading_core.common.model.feature_view_request import FeatureViewRequest


class TestFeatureViewRequest:
    """Test the FeatureViewRequest container pattern."""

    def test_create_valid_request(self, mock_features_list: list[BaseFeature]) -> None:
        """Test creating a valid feature view request."""
        # Given
        feature_version_info = FeatureConfigVersionInfo(
            semver="1.0.0",
            hash="test_hash",
            created_at=datetime.now(),
            feature_definitions=[]
        )

        # When
        request = FeatureViewRequest.create(
            symbol="EURUSD",
            feature_view_name="test_view",
            feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature_version_info=feature_version_info,
            features=mock_features_list
        )

        # Then
        assert request.symbol == "EURUSD"
        assert request.feature_view_name == "test_view"
        assert request.feature_role == FeatureRoleEnum.OBSERVATION_SPACE
        assert request.feature_version_info == feature_version_info
        assert request.features == mock_features_list

    def test_request_validation_empty_symbol(self, mock_features_list: list[BaseFeature]) -> None:
        """Test validation fails for empty symbol."""
        # Given
        feature_version_info = FeatureConfigVersionInfo(
            semver="1.0.0",
            hash="test_hash",
            created_at=datetime.now(),
            feature_definitions=[]
        )

        # When/Then
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            FeatureViewRequest.create(
                symbol="",
                feature_view_name="test_view",
                feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
                feature_version_info=feature_version_info,
                features=mock_features_list
            )

    def test_sanitized_accessors(self, mock_features_list: list[BaseFeature]) -> None:
        """Test sanitized accessor methods."""
        # Given
        feature_version_info = FeatureConfigVersionInfo(
            semver="1.0.0",
            hash="test_hash",
            created_at=datetime.now(),
            feature_definitions=[]
        )

        request = FeatureViewRequest(
            symbol="  EURUSD  ",
            feature_view_name="  test_view  ",
            feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature_version_info=feature_version_info,
            features=mock_features_list
        )

        # When/Then
        assert request.get_sanitized_symbol() == "EURUSD"
        assert request.get_sanitized_feature_view_name() == "test_view"
        assert request.get_role_description() == "observation_space"

    def test_none_feature_role_handling(self, mock_features_list: list[BaseFeature]) -> None:
        """Test handling of None feature role for integration tests."""
        # Given
        feature_version_info = FeatureConfigVersionInfo(
            semver="1.0.0",
            hash="test_hash",
            created_at=datetime.now(),
            feature_definitions=[]
        )

        # When
        request = FeatureViewRequest.create(
            symbol="EURUSD",
            feature_view_name="test_view",
            feature_role=None,
            feature_version_info=feature_version_info,
            features=mock_features_list
        )

        # Then
        assert request.feature_role is None
        assert request.get_role_description() == "None"

    @pytest.mark.parametrize("invalid_role", [
        "not_an_enum",
        123,
        [],
        {}
    ])
    def test_invalid_feature_role_validation(self, invalid_role: Any, mock_features_list: list[BaseFeature]) -> None:
        """Test validation fails for invalid feature role types."""
        # Given
        feature_version_info = FeatureConfigVersionInfo(
            semver="1.0.0",
            hash="test_hash",
            created_at=datetime.now(),
            feature_definitions=[]
        )

        # When/Then
        with pytest.raises(ValueError, match="Feature role must be a FeatureRoleEnum or None"):
            FeatureViewRequest(
                symbol="EURUSD",
                feature_view_name="test_view",
                feature_role=invalid_role,
                feature_version_info=feature_version_info,
                features=mock_features_list
            ).validate()


class TestFeatureViewRequestBenefits:
    """Demonstrate the benefits of the container pattern."""

    def test_readability_comparison(self, mock_features_list: list[BaseFeature]) -> None:
        """
        Demonstrate how the container improves readability.

        Compare:
        OLD: provider.create_feature_view("EURUSD", "test_view", FeatureRoleEnum.OBSERVATION_SPACE, version_info)
        NEW: provider.create_feature_view_from_request(request)
        """
        # Given
        feature_version_info = FeatureConfigVersionInfo(
            semver="1.0.0",
            hash="test_hash",
            created_at=datetime.now(),
            feature_definitions=[]
        )

        # New approach: Self-documenting and clear
        request = FeatureViewRequest.create(
            symbol="EURUSD",
            feature_view_name="observation_features",
            feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature_version_info=feature_version_info,
            features=mock_features_list
        )

        # Then
        # The container makes the intent clear and groups related parameters
        assert request.symbol == "EURUSD"
        assert request.feature_view_name == "observation_features"
        # Parameters are validated as a unit
        # Easy to extend with new parameters without breaking existing calls
        # Better testability - can create test fixtures easily

    def test_extensibility_benefit(self, mock_features_list: list[BaseFeature]) -> None:
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

        feature_version_info = FeatureConfigVersionInfo(
            semver="1.0.0",
            hash="test_hash",
            created_at=datetime.now(),
            feature_definitions=[]
        )

        request = FeatureViewRequest.create(
            symbol="EURUSD",
            feature_view_name="test_view",
            feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature_version_info=feature_version_info,
            features=mock_features_list
        )

        # When/Then
        # This pattern allows for future enhancement without API breaks
        assert request is not None
