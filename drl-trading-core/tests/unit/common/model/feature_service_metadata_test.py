"""
Test for the FeatureServiceRequestContainer dataclass.

This demonstrates the container pattern for feature service request parameters.
"""
from datetime import datetime
from typing import Any

import pytest
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.adapter.model.feature_config_version_info import FeatureConfigVersionInfo
from drl_trading_common.core.model.timeframe import Timeframe

from drl_trading_common.core.model.dataset_identifier import DatasetIdentifier

from drl_trading_core.core.dto.feature_service_metadata import FeatureServiceMetadata


@pytest.fixture
def mock_feature_config_version_info() -> FeatureConfigVersionInfo:
    """Create a mock feature config version info for testing."""
    return FeatureConfigVersionInfo(
        semver="1.0.0",
        hash="abc123def456",
        created_at=datetime(2025, 1, 1, 12, 0, 0),
        feature_definitions=[
            {"name": "sma_20", "type": "technical_indicator"},
            {"name": "rsi_14", "type": "technical_indicator"}
        ],
        description="Test feature configuration"
    )


class TestFeatureServiceRequestContainer:
    """Test the FeatureServiceRequestContainer dataclass."""

    def test_create_valid_request(
        self,
        mock_feature_config_version_info: FeatureConfigVersionInfo,
        mock_timeframe: Timeframe
    ) -> None:
        """Test creating a valid feature service request."""
        # Given
        dataset_identifier = DatasetIdentifier("EURUSD", mock_timeframe)
        feature_role = FeatureRoleEnum.OBSERVATION_SPACE
        feature_view_metadata_list = []  # Empty for this test

        # When
        request = FeatureServiceMetadata.create(
            dataset_identifier=dataset_identifier,
            feature_role=feature_role,
            feature_config_version=mock_feature_config_version_info,
            feature_view_metadata_list=feature_view_metadata_list
        )

        # Then
        assert request.dataset_identifier.symbol == "EURUSD"
        assert request.feature_service_role == FeatureRoleEnum.OBSERVATION_SPACE
        assert request.feature_version_info == mock_feature_config_version_info
        assert request.dataset_identifier.timeframe == mock_timeframe

    def test_direct_instantiation(
        self,
        mock_feature_config_version_info: FeatureConfigVersionInfo,
        mock_timeframe: Timeframe
    ) -> None:
        """Test direct instantiation of FeatureServiceRequestContainer."""
        # Given
        dataset_identifier = DatasetIdentifier("EURUSD", mock_timeframe)
        feature_role = FeatureRoleEnum.OBSERVATION_SPACE
        feature_view_metadata_list = []

        # When
        request = FeatureServiceMetadata(
            dataset_identifier=dataset_identifier,
            feature_service_role=feature_role,
            feature_version_info=mock_feature_config_version_info,
            feature_view_metadata_list=feature_view_metadata_list
        )

        # Then
        assert request.dataset_identifier.symbol == "EURUSD"
        assert request.feature_service_role == FeatureRoleEnum.OBSERVATION_SPACE
        assert request.feature_version_info == mock_feature_config_version_info
        assert request.dataset_identifier.timeframe == mock_timeframe

    def test_validation_empty_symbol(
        self,
        mock_feature_config_version_info: FeatureConfigVersionInfo,
        mock_timeframe: Timeframe
    ) -> None:
        """Test validation fails for empty symbol."""
        # Given
        empty_dataset_identifier = DatasetIdentifier("", mock_timeframe)

        # When/Then
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            FeatureServiceMetadata.create(
                dataset_identifier=empty_dataset_identifier,
                feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
                feature_config_version=mock_feature_config_version_info,
                feature_view_metadata_list=[]
            )

    def test_validation_whitespace_only_symbol(
        self,
        mock_feature_config_version_info: FeatureConfigVersionInfo,
        mock_timeframe: Timeframe
    ) -> None:
        """Test validation fails for whitespace-only symbol."""
        # Given
        whitespace_dataset_identifier = DatasetIdentifier("   ", mock_timeframe)

        # When/Then
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            FeatureServiceMetadata.create(
                dataset_identifier=whitespace_dataset_identifier,
                feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
                feature_config_version=mock_feature_config_version_info,
                feature_view_metadata_list=[]
            )

    def test_validation_none_symbol(
        self,
        mock_feature_config_version_info: FeatureConfigVersionInfo,
        mock_timeframe: Timeframe
    ) -> None:
        """Test validation fails for None symbol."""
        # Given/When/Then
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            FeatureServiceMetadata.create(
                dataset_identifier=DatasetIdentifier(None, mock_timeframe),  # type: ignore[arg-type]
                feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
                feature_config_version=mock_feature_config_version_info,
                feature_view_metadata_list=[]
            )

    @pytest.mark.parametrize("invalid_role", [
        "not_an_enum",
        123,
        [],
        {},
        None
    ])
    def test_validation_invalid_feature_role(
        self,
        invalid_role: Any,
        mock_feature_config_version_info: FeatureConfigVersionInfo,
        mock_timeframe: Timeframe
    ) -> None:
        """Test validation fails for invalid feature role types."""
        # Given
        dataset_identifier = DatasetIdentifier("EURUSD", mock_timeframe)

        # When/Then
        with pytest.raises(ValueError, match="Feature role must be a FeatureRoleEnum"):
            FeatureServiceMetadata.create(
                dataset_identifier=dataset_identifier,
                feature_role=invalid_role,  # type: ignore[arg-type]
                feature_config_version=mock_feature_config_version_info,
                feature_view_metadata_list=[]
            )

    @pytest.mark.parametrize("invalid_feature_version", [
        "not_a_feature_version",
        123,
        [],
        {},
        None
    ])
    def test_validation_invalid_feature_version_info(
        self,
        invalid_feature_version: Any,
        mock_timeframe: Timeframe
    ) -> None:
        """Test validation fails for invalid feature version info types."""
        # Given
        dataset_identifier = DatasetIdentifier("EURUSD", mock_timeframe)
        feature_role = FeatureRoleEnum.OBSERVATION_SPACE

        # When/Then
        with pytest.raises(ValueError, match="Feature must be a FeatureConfigVersionInfo"):
            FeatureServiceMetadata.create(
                dataset_identifier=dataset_identifier,
                feature_role=feature_role,
                feature_config_version=invalid_feature_version,  # type: ignore[arg-type]
                feature_view_metadata_list=[]
            )

    @pytest.mark.parametrize("invalid_timeframe", [
        "not_a_timeframe",
        123,
        [],
        {},
        None
    ])
    def test_validation_invalid_timeframe(
        self,
        invalid_timeframe: Any,
        mock_feature_config_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test validation fails for invalid timeframe types."""
        # Given
        dataset_identifier = DatasetIdentifier("EURUSD", invalid_timeframe)  # type: ignore[arg-type]

        # When/Then
        with pytest.raises(ValueError, match="Timeframe must be a Timeframe"):
            FeatureServiceMetadata.create(
                dataset_identifier=dataset_identifier,
                feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
                feature_config_version=mock_feature_config_version_info,
                feature_view_metadata_list=[]
            )

    def test_get_sanitized_symbol(
        self,
        mock_feature_config_version_info: FeatureConfigVersionInfo,
        mock_timeframe: Timeframe
    ) -> None:
        """Test sanitized symbol accessor method."""
        # Given
        symbol_with_whitespace = "  EURUSD  "
        request = FeatureServiceMetadata(
            dataset_identifier=DatasetIdentifier(symbol_with_whitespace, mock_timeframe),
            feature_service_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature_version_info=mock_feature_config_version_info,
            feature_view_metadata_list=[]
        )

        # When
        sanitized_symbol = request.get_sanitized_symbol()

        # Then
        assert sanitized_symbol == "EURUSD"

    def test_get_sanitized_symbol_empty_string(
        self,
        mock_feature_config_version_info: FeatureConfigVersionInfo,
        mock_timeframe: Timeframe
    ) -> None:
        """Test sanitized symbol returns empty string for None symbol."""
        # Given
        request = FeatureServiceMetadata(
            dataset_identifier=DatasetIdentifier(None, mock_timeframe),  # type: ignore[arg-type]
            feature_service_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature_version_info=mock_feature_config_version_info,
            feature_view_metadata_list=[]
        )

        # When
        sanitized_symbol = request.get_sanitized_symbol()

        # Then
        assert sanitized_symbol == ""

    def test_validation_called_on_create(
        self,
        mock_feature_config_version_info: FeatureConfigVersionInfo,
        mock_timeframe: Timeframe
    ) -> None:
        """Test that validation is called when using create factory method."""
        # Given
        valid_dataset_identifier = DatasetIdentifier("EURUSD", mock_timeframe)
        valid_role = FeatureRoleEnum.OBSERVATION_SPACE

        # When
        request = FeatureServiceMetadata.create(
            dataset_identifier=valid_dataset_identifier,
            feature_role=valid_role,
            feature_config_version=mock_feature_config_version_info,
            feature_view_metadata_list=[]
        )

        # Then
        # Should not raise any exception
        assert request is not None
        assert request.dataset_identifier.symbol == "EURUSD"
        assert request.feature_service_role == valid_role
        assert request.feature_version_info == mock_feature_config_version_info
        assert request.dataset_identifier.timeframe == mock_timeframe

    def test_manual_validation_call(
        self,
        mock_feature_config_version_info: FeatureConfigVersionInfo,
        mock_timeframe: Timeframe
    ) -> None:
        """Test calling validation manually on a valid request."""
        # Given
        request = FeatureServiceMetadata(
            dataset_identifier=DatasetIdentifier("EURUSD", mock_timeframe),
            feature_service_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature_version_info=mock_feature_config_version_info,
            feature_view_metadata_list=[]
        )

        # When/Then
        # Should not raise any exception
        request.validate()


class TestFeatureServiceRequestContainerBenefits:
    """Demonstrate the benefits of the container pattern."""

    def test_readability_improvement(
        self,
        mock_feature_config_version_info: FeatureConfigVersionInfo,
        mock_timeframe: Timeframe
    ) -> None:
        """
        Demonstrate how the container improves readability.

        Compare:
        OLD: service.request_features("EURUSD", FeatureRoleEnum.OBSERVATION_SPACE, version_info, timeframe)
        NEW: service.request_features_from_container(request)
        """
        # Given
        # New approach: Self-documenting and clear
        request = FeatureServiceMetadata.create(
            dataset_identifier=DatasetIdentifier("EURUSD", mock_timeframe),
            feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature_config_version=mock_feature_config_version_info,
            feature_view_metadata_list=[]
        )

        # Then
        # The container makes the intent clear and groups related parameters
        assert request.dataset_identifier.symbol == "EURUSD"
        assert request.feature_service_role == FeatureRoleEnum.OBSERVATION_SPACE
        assert request.feature_version_info == mock_feature_config_version_info
        assert request.dataset_identifier.timeframe == mock_timeframe
        # Parameters are validated as a unit
        # Better testability - can create test fixtures easily

    def test_extensibility_benefit(
        self,
        mock_feature_config_version_info: FeatureConfigVersionInfo,
        mock_timeframe: Timeframe
    ) -> None:
        """
        Demonstrate how the container makes the API more extensible.

        Adding new parameters to the container doesn't break existing code.
        """
        # Given - we could easily add new fields to FeatureServiceRequestContainer:
        # - cache_enabled: bool = True
        # - priority: int = 0
        # - metadata: Dict[str, Any] = field(default_factory=dict)
        #
        # Without breaking any existing calls!

        request = FeatureServiceMetadata.create(
            dataset_identifier=DatasetIdentifier("EURUSD", mock_timeframe),
            feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature_config_version=mock_feature_config_version_info,
            feature_view_metadata_list=[]
        )

        # When/Then
        # This pattern allows for future enhancement without API breaks
        assert request is not None

    def test_parameter_grouping_benefit(
        self,
        mock_feature_config_version_info: FeatureConfigVersionInfo,
        mock_timeframe: Timeframe
    ) -> None:
        """Demonstrate how the container groups related parameters logically."""
        # Given
        request = FeatureServiceMetadata.create(
            dataset_identifier=DatasetIdentifier("EURUSD", mock_timeframe),
            feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature_config_version=mock_feature_config_version_info,
            feature_view_metadata_list=[]
        )

        # When/Then
        # All feature service related parameters are grouped together
        # This makes it easier to:
        # 1. Pass around as a single unit
        # 2. Validate as a cohesive set
        # 3. Test with consistent fixtures
        # 4. Extend without breaking existing interfaces
        assert request.dataset_identifier.symbol == "EURUSD"
        assert request.dataset_identifier.timeframe == mock_timeframe
        assert request.feature_service_role == FeatureRoleEnum.OBSERVATION_SPACE
        assert request.feature_version_info == mock_feature_config_version_info
