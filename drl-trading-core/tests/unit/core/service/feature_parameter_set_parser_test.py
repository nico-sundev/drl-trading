"""Unit tests for the FeatureParameterSetParser class."""

import pytest
from unittest.mock import Mock

from drl_trading_core.core.service.feature.feature_factory_interface import IFeatureFactory
from drl_trading_core.core.model.feature_definition import FeatureDefinition
from drl_trading_core.core.service.feature_parameter_set_parser import FeatureParameterSetParser


class TestFeatureParameterSetParser:
    """Test cases for FeatureParameterSetParser class."""

    @pytest.fixture
    def mock_feature_factory(self) -> Mock:
        """Create mock for IFeatureFactory."""
        return Mock(spec=IFeatureFactory)

    @pytest.fixture
    def parser(self, mock_feature_factory: Mock) -> FeatureParameterSetParser:
        """Create FeatureParameterSetParser instance with mocked dependencies."""
        return FeatureParameterSetParser(feature_factory=mock_feature_factory)

    @pytest.fixture
    def mock_feature_definition(self) -> Mock:
        """Create mock FeatureDefinition."""
        feature_def = Mock(spec=FeatureDefinition)
        feature_def.name = "rsi"
        feature_def.raw_parameter_sets = [{"period": 14}]
        feature_def.parsed_parameter_sets = {}
        return feature_def

    def test_initialization(self, mock_feature_factory: Mock) -> None:
        """Test FeatureParameterSetParser initialization."""
        # Given & When
        parser = FeatureParameterSetParser(feature_factory=mock_feature_factory)

        # Then
        assert parser.feature_factory is mock_feature_factory

    def test_parse_parameter_set_success(
        self, parser: FeatureParameterSetParser, mock_feature_definition: Mock, mock_feature_factory: Mock
    ) -> None:
        """Test parse_parameter_set with successful config creation."""
        # Given
        raw_param_set = {"period": 14}
        mock_config = Mock()
        mock_config.hash_id.return_value = "hash123"
        mock_feature_factory.create_config_instance.return_value = mock_config

        # When
        parser.parse_parameter_set(mock_feature_definition, raw_param_set)

        # Then
        mock_feature_factory.create_config_instance.assert_called_once_with("rsi", raw_param_set)
        assert mock_feature_definition.parsed_parameter_sets["hash123"] is mock_config

    def test_parse_parameter_set_none_config(
        self, parser: FeatureParameterSetParser, mock_feature_definition: Mock, mock_feature_factory: Mock
    ) -> None:
        """Test parse_parameter_set when config instance is None."""
        # Given
        raw_param_set = {"period": 14}
        mock_feature_factory.create_config_instance.return_value = None

        # When
        parser.parse_parameter_set(mock_feature_definition, raw_param_set)

        # Then
        mock_feature_factory.create_config_instance.assert_called_once_with("rsi", raw_param_set)
        assert len(mock_feature_definition.parsed_parameter_sets) == 0

    def test_parse_parameter_set_duplicate(
        self, parser: FeatureParameterSetParser, mock_feature_definition: Mock, mock_feature_factory: Mock
    ) -> None:
        """Test parse_parameter_set with duplicate hash_id."""
        # Given
        raw_param_set = {"period": 14}
        mock_config = Mock()
        mock_config.hash_id.return_value = "hash123"
        mock_feature_factory.create_config_instance.return_value = mock_config
        mock_feature_definition.parsed_parameter_sets = {"hash123": Mock()}  # Already exists

        # When
        parser.parse_parameter_set(mock_feature_definition, raw_param_set)

        # Then
        mock_feature_factory.create_config_instance.assert_called_once_with("rsi", raw_param_set)
        # Should not add duplicate
        assert len(mock_feature_definition.parsed_parameter_sets) == 1

    def test_parse_feature_definitions(
        self, parser: FeatureParameterSetParser, mock_feature_factory: Mock
    ) -> None:
        """Test parse_feature_definitions with multiple feature definitions."""
        # Given
        feature_def1 = Mock(spec=FeatureDefinition)
        feature_def1.name = "rsi"
        feature_def1.raw_parameter_sets = [{"period": 14}, {"period": 21}]
        feature_def1.parsed_parameter_sets = {}

        feature_def2 = Mock(spec=FeatureDefinition)
        feature_def2.name = "close_price"
        feature_def2.raw_parameter_sets = []
        feature_def2.parsed_parameter_sets = {}

        mock_config1 = Mock()
        mock_config1.hash_id.return_value = "hash123"
        mock_config2 = Mock()
        mock_config2.hash_id.return_value = "hash456"
        mock_feature_factory.create_config_instance.side_effect = [mock_config1, mock_config2]

        # When
        parser.parse_feature_definitions([feature_def1, feature_def2])

        # Then
        assert mock_feature_factory.create_config_instance.call_count == 2
        assert len(feature_def1.parsed_parameter_sets) == 2

    def test_parse_feature_definitions_empty_list(
        self, parser: FeatureParameterSetParser
    ) -> None:
        """Test parse_feature_definitions with empty list."""
        # Given & When
        parser.parse_feature_definitions([])

        # Then
        # No calls expected

    def test_parse_feature_definitions_invalid_param_type(
        self, parser: FeatureParameterSetParser, mock_feature_factory: Mock
    ) -> None:
        """Test parse_feature_definitions with invalid parameter set type."""
        # Given
        feature_def = Mock(spec=FeatureDefinition)
        feature_def.name = "rsi"
        feature_def.raw_parameter_sets = ["invalid"]  # Not dict
        feature_def.parsed_parameter_sets = {}

        # When & Then
        with pytest.raises(ValueError, match="Invalid parameter set: Expected a dictionary"):
            parser.parse_feature_definitions([feature_def])

    def test_parse_parameter_set_empty_dict(
        self, parser: FeatureParameterSetParser, mock_feature_definition: Mock, mock_feature_factory: Mock
    ) -> None:
        """Test parse_parameter_set with empty parameter dict."""
        # Given
        raw_param_set = {}
        mock_config = Mock()
        mock_config.hash_id.return_value = "hash_empty"
        mock_feature_factory.create_config_instance.return_value = mock_config

        # When
        parser.parse_parameter_set(mock_feature_definition, raw_param_set)

        # Then
        mock_feature_factory.create_config_instance.assert_called_once_with("rsi", raw_param_set)
        assert mock_feature_definition.parsed_parameter_sets["hash_empty"] is mock_config

    def test_parse_parameter_set_parsed_sets_none(
        self, parser: FeatureParameterSetParser, mock_feature_factory: Mock
    ) -> None:
        """Test parse_parameter_set when parsed_parameter_sets is None."""
        # Given
        feature_def = Mock(spec=FeatureDefinition)
        feature_def.name = "rsi"
        feature_def.parsed_parameter_sets = None  # None instead of dict
        raw_param_set = {"period": 14}
        mock_config = Mock()
        mock_config.hash_id.return_value = "hash123"
        mock_feature_factory.create_config_instance.return_value = mock_config

        # When & Then
        with pytest.raises(AttributeError):
            parser.parse_parameter_set(feature_def, raw_param_set)

    def test_parse_parameter_set_factory_exception(
        self, parser: FeatureParameterSetParser, mock_feature_definition: Mock, mock_feature_factory: Mock
    ) -> None:
        """Test parse_parameter_set when factory raises exception."""
        # Given
        raw_param_set = {"period": 14}
        mock_feature_factory.create_config_instance.side_effect = ValueError("Invalid config")

        # When & Then
        with pytest.raises(ValueError, match="Invalid config"):
            parser.parse_parameter_set(mock_feature_definition, raw_param_set)

    def test_parse_feature_definitions_with_none_in_list(
        self, parser: FeatureParameterSetParser, mock_feature_factory: Mock
    ) -> None:
        """Test parse_feature_definitions with None in raw_parameter_sets."""
        # Given
        feature_def = Mock(spec=FeatureDefinition)
        feature_def.name = "rsi"
        feature_def.raw_parameter_sets = [None]  # None in list
        feature_def.parsed_parameter_sets = {}

        # When & Then
        with pytest.raises(ValueError, match="Invalid parameter set: Expected a dictionary"):
            parser.parse_feature_definitions([feature_def])

    def test_parse_feature_definitions_name_none(
        self, parser: FeatureParameterSetParser, mock_feature_factory: Mock
    ) -> None:
        """Test parse_feature_definitions with feature name as None."""
        # Given
        feature_def = Mock(spec=FeatureDefinition)
        feature_def.name = None  # None name
        feature_def.raw_parameter_sets = [{"period": 14}]
        feature_def.parsed_parameter_sets = {}

        # When & Then
        with pytest.raises(ValueError, match="Feature name is required"):
            parser.parse_feature_definitions([feature_def])
