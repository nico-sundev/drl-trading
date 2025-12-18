"""Unit tests for FeatureFieldMapper."""

from unittest.mock import Mock

from feast.types import Float32
from drl_trading_adapter.adapter.feature_store.provider.mapper.feature_field_mapper import FeatureFieldFactory
from drl_trading_core.core.model.feature.feature_metadata import FeatureMetadata


class TestFeatureFieldMapper:
    """Unit tests for the FeatureFieldMapper class."""
    def test_create_fields_single_feature(self) -> None:
        """Test field creation for feature without sub-features."""
        # Given
        mapper = FeatureFieldFactory()
        mock_metadata = Mock(spec=FeatureMetadata)
        mock_metadata.feature_name = "close_price"
        mock_metadata.config = None
        mock_metadata.sub_feature_names = []

        # When
        fields = mapper.create_fields(mock_metadata)

        # Then
        assert len(fields) == 1
        assert fields[0].name == "close_price"
        assert fields[0].dtype == Float32

    def test_create_fields_with_sub_features(self) -> None:
        """Test field creation for feature with sub-features."""
        # Given
        mapper = FeatureFieldFactory()
        mock_metadata = Mock(spec=FeatureMetadata)
        mock_metadata.feature_name = "reward"
        mock_metadata.config = None
        mock_metadata.sub_feature_names = [
            "reward",
            "cumulative_return",
        ]

        # When
        fields = mapper.create_fields(mock_metadata)

        # Then
        assert len(fields) == 2
        field_names = [field.name for field in fields]
        assert "reward_reward" in field_names
        assert "reward_cumulative_return" in field_names
        for field in fields:
            assert field.dtype == Float32

    def test_create_fields_with_config_and_sub_features(self) -> None:
        """Test field creation for feature with both config and sub-features."""
        # Given
        mapper = FeatureFieldFactory()
        mock_metadata = Mock(spec=FeatureMetadata)
        mock_metadata.feature_name = "complex_indicator"
        mock_metadata.config_to_string = "param1"
        mock_metadata.sub_feature_names = ["upper", "lower"]

        mock_config = Mock()
        mock_config.hash_id.return_value = "hash123"
        mock_metadata.config = mock_config

        # When
        fields = mapper.create_fields(mock_metadata)

        # Then
        assert len(fields) == 2
        field_names = [field.name for field in fields]
        assert "complex_indicator_param1_hash123_upper" in field_names
        assert "complex_indicator_param1_hash123_lower" in field_names
