import pytest
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum

from drl_trading_core.preprocess.feature_store.mapper.feature_view_name_mapper import (
    FeatureViewNameMapper,
)
from drl_trading_core.preprocess.feature_store.feature_view_name_enum import (
    FeatureViewNameEnum,
)


class TestFeatureViewNameMapper:
    """Test cases for FeatureViewNameMapper class."""

    @pytest.fixture
    def mapper(self) -> FeatureViewNameMapper:
        """Fixture to provide a FeatureViewNameMapper instance."""
        return FeatureViewNameMapper()

    def test_map_observation_space_role(self, mapper: FeatureViewNameMapper) -> None:
        """Test mapping of OBSERVATION_SPACE feature role to correct feature view name."""
        # Given
        feature_role = FeatureRoleEnum.OBSERVATION_SPACE
        expected_view_name = FeatureViewNameEnum.OBSERVATION_SPACE.value

        # When
        result = mapper.map(feature_role)

        # Then
        assert result == expected_view_name

    def test_map_reward_engineering_role(self, mapper: FeatureViewNameMapper) -> None:
        """Test mapping of REWARD_ENGINEERING feature role to correct feature view name."""
        # Given
        feature_role = FeatureRoleEnum.REWARD_ENGINEERING
        expected_view_name = FeatureViewNameEnum.REWARD_ENGINEERING.value

        # When
        result = mapper.map(feature_role)

        # Then
        assert result == expected_view_name

    @pytest.mark.parametrize(
        "feature_role,expected_view_name",
        [
            (FeatureRoleEnum.OBSERVATION_SPACE, FeatureViewNameEnum.OBSERVATION_SPACE.value),
            (FeatureRoleEnum.REWARD_ENGINEERING, FeatureViewNameEnum.REWARD_ENGINEERING.value),
        ],
    )
    def test_map_all_valid_roles(
        self,
        mapper: FeatureViewNameMapper,
        feature_role: FeatureRoleEnum,
        expected_view_name: str
    ) -> None:
        """Test mapping for all valid feature roles using parametrized test."""
        # Given
        # feature_role and expected_view_name are provided by parametrize

        # When
        result = mapper.map(feature_role)

        # Then
        assert result == expected_view_name

    def test_map_invalid_role_raises_value_error(self, mapper: FeatureViewNameMapper) -> None:
        """Test that mapping an invalid feature role raises ValueError."""
        # Given
        # Create a mock feature role that doesn't exist in our enum
        # We'll use a string that's not a valid enum value
        invalid_role = "INVALID_ROLE"

        # When & Then
        with pytest.raises(ValueError, match=f"Unknown feature role: {invalid_role}"):
            mapper.map(invalid_role)

    def test_map_none_role_raises_error(self, mapper: FeatureViewNameMapper) -> None:
        """Test that mapping None feature role raises appropriate error."""
        # Given
        none_role = None

        # When & Then
        with pytest.raises((ValueError, AttributeError)):
            mapper.map(none_role)

    def test_mapper_is_stateless(self, mapper: FeatureViewNameMapper) -> None:
        """Test that mapper is stateless and multiple calls return consistent results."""
        # Given
        feature_role = FeatureRoleEnum.OBSERVATION_SPACE
        expected_view_name = FeatureViewNameEnum.OBSERVATION_SPACE.value

        # When
        result1 = mapper.map(feature_role)
        result2 = mapper.map(feature_role)
        result3 = mapper.map(feature_role)

        # Then
        assert result1 == expected_view_name
        assert result2 == expected_view_name
        assert result3 == expected_view_name
        assert result1 == result2 == result3

    def test_mapper_handles_all_enum_values(self, mapper: FeatureViewNameMapper) -> None:
        """Test that mapper can handle all existing FeatureRoleEnum values."""
        # Given
        all_feature_roles = list(FeatureRoleEnum)

        # When & Then
        for feature_role in all_feature_roles:
            # Should not raise any exception for valid enum values
            result = mapper.map(feature_role)
            # Result should be a valid FeatureViewNameEnum value
            assert isinstance(result, str)
            assert result in [view.value for view in FeatureViewNameEnum]


class TestFeatureViewNameMapperEdgeCases:
    """Test edge cases and error conditions for FeatureViewNameMapper."""

    @pytest.fixture
    def mapper(self) -> FeatureViewNameMapper:
        """Fixture to provide a FeatureViewNameMapper instance."""
        return FeatureViewNameMapper()

    def test_map_with_mock_enum_value(self, mapper: FeatureViewNameMapper, monkeypatch) -> None:
        """Test behavior when a new enum value is added but not handled in mapper."""
        # Given
        # This test simulates what happens if someone adds a new enum value
        # but forgets to update the mapper

        # Create a new mock enum value (simulating future enum expansion)
        class MockFeatureRole:
            def __init__(self, value: str):
                self.value = value

            def __eq__(self, other):
                return False  # This will not match any existing enum values

            def __str__(self):
                return f"MockFeatureRole({self.value})"

        mock_role = MockFeatureRole("NEW_FEATURE_ROLE")

        # When & Then
        with pytest.raises(ValueError, match="Unknown feature role"):
            mapper.map(mock_role)

    def test_mapper_return_types(self, mapper: FeatureViewNameMapper) -> None:
        """Test that mapper returns correct types for all valid inputs."""
        # Given
        test_cases = [
            FeatureRoleEnum.OBSERVATION_SPACE,
            FeatureRoleEnum.REWARD_ENGINEERING,
        ]

        # When & Then
        for feature_role in test_cases:
            result = mapper.map(feature_role)

            # Should return a string (enum value)
            assert isinstance(result, str)
            # Should be a valid FeatureViewNameEnum value
            assert result in [view.value for view in FeatureViewNameEnum]
