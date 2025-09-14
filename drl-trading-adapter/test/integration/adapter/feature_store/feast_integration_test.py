
from unittest.mock import Mock
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_core.common.model.feature_view_request import FeatureViewRequestContainer
from feast import FeatureStore, Field
from feast.types import Float32
from injector import Injector
from drl_trading_common.base import BaseFeature

from drl_trading_adapter.adapter.feature_store.provider import (
    FeastProvider,
)
from drl_trading_adapter.adapter.feature_store.provider import (
    FeatureStoreWrapper,
)
from drl_trading_adapter.adapter.feature_store.provider.mapper import (
    IFeatureFieldMapper,
)


class _TestFeature(BaseFeature):
    """Minimal test implementation of BaseFeature for integration tests."""

    def __init__(self, name: str):
        # Skip parent constructor to avoid complex dependencies
        self.name = name
        self.config = None

    def get_feature_name(self) -> str:
        return self.name

    def get_sub_features_names(self) -> list[str]:
        return []

    def get_config_to_string(self) -> str | None:
        return None

    def get_config(self):
        return None

    def compute_all(self):
        # Not used in integration tests
        return None

    def add(self, df):
        # Not used in integration tests
        pass

    def compute_latest(self):
        # Not used in integration tests
        return None


def create_mock_mapper(field_responses: dict[str, list[Field]]) -> Mock:
    """Create a mock feature field mapper that returns predefined fields."""
    mock_mapper = Mock(spec=IFeatureFieldMapper)

    def side_effect_create_fields(feature):
        feature_name = feature.get_feature_name()
        return field_responses.get(feature_name, [Field(name=feature_name, dtype=Float32)])

    def side_effect_get_field_base_name(feature):
        return feature.get_feature_name()

    mock_mapper.create_fields.side_effect = side_effect_create_fields
    mock_mapper.get_field_base_name.side_effect = side_effect_get_field_base_name
    return mock_mapper


def create_simple_mock_feature(name: str) -> _TestFeature:
    """Create a TestFeature instance for integration tests."""
    return _TestFeature(name)


class TestFeastIntegration:
    """Integration tests for Feast feature store functionality."""

    def test_feature_store_wrapper_provides_real_feature_store(
        self,
        real_feast_container: Injector,
        temp_feast_repo: str,
        clean_integration_environment: None
    ) -> None:
        """Test that FeatureStoreWrapper provides a real FeatureStore instance."""
        # Given
        feature_store_wrapper = real_feast_container.get(FeatureStoreWrapper)

        # When
        feature_store = feature_store_wrapper.get_feature_store()

        # Then
        assert isinstance(feature_store, FeatureStore)
        assert feature_store is not None

        # Verify it's properly configured for test environment
        assert feature_store.repo_path is not None

    def test_feast_provider_can_create_feature_views_with_request(
        self,
        real_feast_container: Injector,
        feature_version_info_fixture,
        temp_feast_repo: str,
        clean_integration_environment: None
    ) -> None:
        """Test that FeastProvider can create feature views using synthetic FeatureViewRequest."""
        # Given
        feast_provider = real_feast_container.get(FeastProvider)

        # Verify that we're using a real Feast store, not a mock
        assert isinstance(feast_provider.get_feature_store(), FeatureStore)

        # Create synthetic features for the request (no complex business logic needed)
        mock_feature_1 = create_simple_mock_feature("test_feature_1")
        mock_feature_2 = create_simple_mock_feature("test_feature_2")

        # Create a mock mapper that returns specific fields for these features
        expected_fields = {
            "test_feature_1": [Field(name="test_feature_1", dtype=Float32)],
            "test_feature_2": [Field(name="test_feature_2", dtype=Float32)],
        }
        mock_mapper = create_mock_mapper(expected_fields)

        # Replace the mapper in the DI container with our mock
        feast_provider.feature_field_mapper = mock_mapper

        # Create FeatureViewRequest with synthetic features
        request_1 = FeatureViewRequestContainer(
            symbol="TESTSYM",
            feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature=mock_feature_1
        )

        request_2 = FeatureViewRequestContainer(
            symbol="TESTSYM",
            feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature=mock_feature_2
        )

        # When
        feature_views = feast_provider._process_feature_view_creation_requests([request_1, request_2])

        # Then
        assert feature_views is not None
        assert len(feature_views) == 2

        # Get the first feature view for detailed assertions
        feature_view = feature_views[0]
        assert "TESTSYM" in [tag for tag in feature_view.tags.values()]

        # Verify fields were created correctly from mock mapper
        all_field_names = []
        for fv in feature_views:
            all_field_names.extend([field.name for field in fv.schema])
        assert "test_feature_1" in all_field_names
        assert "test_feature_2" in all_field_names

    def test_feast_provider_uses_injected_mapper(
        self,
        real_feast_container: Injector,
        feature_version_info_fixture,
        temp_feast_repo: str,
        clean_integration_environment: None
    ) -> None:
        """Test that FeastProvider uses the injected feature field mapper correctly."""
        # Given
        feast_provider = real_feast_container.get(FeastProvider)

        # Create simple mock features
        mock_feature = create_simple_mock_feature("mapper_test_feature")

        # Create a mock mapper with custom behavior
        mock_mapper = Mock(spec=IFeatureFieldMapper)
        mock_mapper.create_fields.return_value = [
            Field(name="custom_mapped_field", dtype=Float32)
        ]
        mock_mapper.get_field_base_name.return_value = "mapper_test_feature"

        # Inject the mock mapper
        feast_provider.feature_field_mapper = mock_mapper

        request = FeatureViewRequestContainer(
            symbol="MAPPERTEST",
            feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature=mock_feature
        )

        # When
        feature_views = feast_provider._process_feature_view_creation_requests([request])

        # Then
        assert feature_views is not None
        assert len(feature_views) == 1
        feature_view = feature_views[0]
        field_names = [field.name for field in feature_view.schema]
        assert "custom_mapped_field" in field_names

        # Verify the mapper was called with our feature
        mock_mapper.create_fields.assert_called_once_with(mock_feature)


class TestFeastDataPersistence:
    """Tests for Feast integration mechanics, not data computation."""

    def test_feast_provider_handles_feature_view_request_validation(
        self,
        real_feast_container: Injector,
        clean_integration_environment: None,
        feature_version_info_fixture,
    ) -> None:
        """Test that FeastProvider validates FeatureViewRequest properly."""
        # Given
        feast_provider = real_feast_container.get(FeastProvider)

        # Create a valid request
        valid_request = FeatureViewRequestContainer(
            symbol="TESTSYM",
            feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature=create_simple_mock_feature("test_feature")
        )

        # When & Then - should not raise any validation errors
        feature_views = feast_provider._process_feature_view_creation_requests([valid_request])
        assert feature_views is not None
        assert len(feature_views) == 1

    def test_feast_feature_view_has_correct_metadata(
        self,
        real_feast_container: Injector,
        clean_integration_environment: None,
        feature_version_info_fixture,
    ) -> None:
        """Test that created feature views have correct Feast metadata."""
        # Given
        feast_provider = real_feast_container.get(FeastProvider)
        request = FeatureViewRequestContainer(
            symbol="METADATA_TEST",
            feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature=create_simple_mock_feature("meta_feature")
        )

        # When
        feature_views = feast_provider._process_feature_view_creation_requests([request])

        # Then - verify Feast-specific metadata
        assert feature_views is not None
        assert len(feature_views) == 1
        feature_view = feature_views[0]
        assert len(feature_view.entities) == 1
        # Note: Feast stores entity names as strings in feature_view.entities
        assert feature_view.entities[0] == "test_entity"  # This is the entity name from config
        assert feature_view.ttl is not None
        assert feature_view.ttl.days > 0  # From config
        assert "METADATA_TEST" in feature_view.tags.values()
        assert feature_view.source.timestamp_field == "event_timestamp"
