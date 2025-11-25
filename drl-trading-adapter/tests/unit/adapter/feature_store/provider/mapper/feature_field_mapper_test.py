"""Unit tests for FeatureFieldMapper."""

from unittest.mock import Mock

import pytest
from feast.types import Float32
from drl_trading_adapter.adapter.feature_store.provider.mapper.feature_field_mapper import FeatureFieldMapper
from drl_trading_common.core.model.feature_metadata import FeatureMetadata


class TestFeatureFieldMapper:
    """Unit tests for the FeatureFieldMapper class."""

    def test_get_field_base_name_without_config(self) -> None:
        """Test field name generation for feature without config."""
        # Given
        mapper = FeatureFieldMapper()
        mock_metadata = Mock(spec=FeatureMetadata)
        mock_metadata.feature_name = "simple_feature"
        mock_metadata.config = None

        # When
        field_name = mapper.get_field_base_name(mock_metadata)

        # Then
        assert field_name == "simple_feature"

    def test_get_field_base_name_with_config(self) -> None:
        """Test field name generation for feature with config."""
        # Given
        mapper = FeatureFieldMapper()
        mock_metadata = Mock(spec=FeatureMetadata)
        mock_metadata.feature_name = "rsi"
        mock_metadata.config_to_string = "14"
        mock_config = Mock()
        mock_config.hash_id.return_value = "abc123"
        mock_metadata.config = mock_config

        # When
        field_name = mapper.get_field_base_name(mock_metadata)

        # Then
        assert field_name == "rsi_14_abc123"

    def test_create_fields_single_feature(self) -> None:
        """Test field creation for feature without sub-features."""
        # Given
        mapper = FeatureFieldMapper()
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
        mapper = FeatureFieldMapper()
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
        mapper = FeatureFieldMapper()
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

    @pytest.mark.parametrize(
        "feature_name,config_string,hash_value,expected",
        [
            ("simple", None, None, "simple"),
            ("rsi", "14", "abc123", "rsi_14_abc123"),
            ("macd", "fast12_slow26", "def456", "macd_fast12_slow26_def456"),
        ],
    )
    def test_get_field_base_name_parametrized(
        self, feature_name: str, config_string: str, hash_value: str, expected: str
    ) -> None:
        """Test parameterized field name generation."""
        # Given
        mapper = FeatureFieldMapper()
        mock_metadata = Mock(spec=FeatureMetadata)
        mock_metadata.feature_name = feature_name
        mock_metadata.config_to_string = config_string

        if config_string:
            mock_config = Mock()
            mock_config.hash_id.return_value = hash_value
            mock_metadata.config = mock_config
        else:
            mock_metadata.config = None

        # When
        field_name = mapper.get_field_base_name(mock_metadata)

        # Then
        assert field_name == expected
