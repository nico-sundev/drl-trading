"""
Unit tests for config utils module.

Tests parameter parsing functionality for feature definitions,
including edge cases and error handling scenarios.
"""
import pytest
from unittest.mock import Mock

from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_core.core.model.feature_definition import FeatureDefinition
from drl_trading_common.interface.feature.feature_factory_interface import IFeatureFactory
from drl_trading_core.core.service.feature_parameter_set_parser import FeatureParameterSetParser


class TestParseParameters:
    """Test cases for parse_parameters function."""

    def test_parse_parameters_success(self) -> None:
        """Test successful parameter parsing."""
        # Given
        mock_feature_factory = Mock(spec=IFeatureFactory)
        mock_config_instance = Mock(spec=BaseParameterSetConfig)
        mock_config_instance.hash_id.return_value = "test_hash_123"
        mock_feature_factory.create_config_instance.return_value = mock_config_instance

        parser = FeatureParameterSetParser(mock_feature_factory)
        feature_def = FeatureDefinition(
            name="rsi",
            enabled=True,
            derivatives=[14],
            raw_parameter_sets=[{"period": 14}],
            parsed_parameter_sets={}
        )

        # When
        parser.parse_feature_definitions([feature_def])

        # Then
        mock_feature_factory.create_config_instance.assert_called_once_with("rsi", {"period": 14})
        assert "test_hash_123" in feature_def.parsed_parameter_sets
        assert feature_def.parsed_parameter_sets["test_hash_123"] == mock_config_instance

    def test_parse_parameters_empty_feature_name(self) -> None:
        """Test parsing with empty feature name raises ValueError."""
        # Given
        mock_feature_factory = Mock(spec=IFeatureFactory)
        parser = FeatureParameterSetParser(mock_feature_factory)
        feature_def = FeatureDefinition(
            name="",  # Empty name
            enabled=True,
            derivatives=[],
            raw_parameter_sets=[{"period": 14}],
            parsed_parameter_sets={}
        )

        # When & Then
        with pytest.raises(ValueError, match="Feature name is required"):
            parser.parse_feature_definitions([feature_def])

    def test_parse_parameters_no_parameter_sets(self) -> None:
        """Test parsing with no parameter sets."""
        # Given
        mock_feature_factory = Mock(spec=IFeatureFactory)
        parser = FeatureParameterSetParser(mock_feature_factory)
        feature_def = FeatureDefinition(
            name="close_price",
            enabled=True,
            derivatives=[],
            raw_parameter_sets=[],
            parsed_parameter_sets={}
        )

        # When
        parser.parse_feature_definitions([feature_def])

        # Then
        mock_feature_factory.create_config_instance.assert_not_called()
        assert len(feature_def.parsed_parameter_sets) == 0

    def test_parse_parameters_factory_returns_none(self) -> None:
        """Test parsing when factory returns None for config instance."""
        # Given
        mock_feature_factory = Mock(spec=IFeatureFactory)
        mock_feature_factory.create_config_instance.return_value = None
        parser = FeatureParameterSetParser(mock_feature_factory)

        feature_def = FeatureDefinition(
            name="rsi",
            enabled=True,
            derivatives=[14],
            raw_parameter_sets=[{"period": 14}],
            parsed_parameter_sets={}
        )

        # When & Then
        with pytest.raises(ValueError, match="Failed to parse any parameter sets for feature 'rsi'"):
            parser.parse_feature_definitions([feature_def])

    def test_parse_parameters_multiple_parameter_sets(self) -> None:
        """Test parsing multiple parameter sets."""
        # Given
        mock_feature_factory = Mock(spec=IFeatureFactory)

        config1 = Mock(spec=BaseParameterSetConfig)
        config1.hash_id.return_value = "hash_1"
        config2 = Mock(spec=BaseParameterSetConfig)
        config2.hash_id.return_value = "hash_2"

        mock_feature_factory.create_config_instance.side_effect = [config1, config2]
        parser = FeatureParameterSetParser(mock_feature_factory)

        feature_def = FeatureDefinition(
            name="rsi",
            enabled=True,
            derivatives=[14],
            raw_parameter_sets=[{"period": 14}, {"period": 21}],
            parsed_parameter_sets={}
        )

        # When
        parser.parse_feature_definitions([feature_def])

        # Then
        assert len(feature_def.parsed_parameter_sets) == 2
        assert feature_def.parsed_parameter_sets["hash_1"] == config1
        assert feature_def.parsed_parameter_sets["hash_2"] == config2

    def test_parse_parameters_duplicate_hash_ids(self) -> None:
        """Test parsing with duplicate hash IDs (should not add duplicates)."""
        # Given
        mock_feature_factory = Mock(spec=IFeatureFactory)

        config1 = Mock(spec=BaseParameterSetConfig)
        config1.hash_id.return_value = "same_hash"
        config2 = Mock(spec=BaseParameterSetConfig)
        config2.hash_id.return_value = "same_hash"  # Same hash as config1

        mock_feature_factory.create_config_instance.side_effect = [config1, config2]
        parser = FeatureParameterSetParser(mock_feature_factory)

        feature_def = FeatureDefinition(
            name="rsi",
            enabled=True,
            derivatives=[14],
            raw_parameter_sets=[{"period": 14}, {"period": 14}],  # Same parameters
            parsed_parameter_sets={}
        )

        # When
        parser.parse_feature_definitions([feature_def])

        # Then
        assert len(feature_def.parsed_parameter_sets) == 1
        assert feature_def.parsed_parameter_sets["same_hash"] == config1


class TestParseAllParameters:
    """Test cases for parse_all_parameters function."""

    def test_parse_all_parameters_success(self) -> None:
        """Test successful parsing of multiple feature definitions."""
        # Given
        mock_feature_factory = Mock(spec=IFeatureFactory)
        mock_config_instance = Mock(spec=BaseParameterSetConfig)
        mock_config_instance.hash_id.return_value = "test_hash"
        mock_feature_factory.create_config_instance.return_value = mock_config_instance

        parser = FeatureParameterSetParser(mock_feature_factory)
        feature_defs = [
            FeatureDefinition(
                name="rsi",
                enabled=True,
                derivatives=[14],
                raw_parameter_sets=[{"period": 14}],
                parsed_parameter_sets={}
            ),
            FeatureDefinition(
                name="sma",
                enabled=True,
                derivatives=[20],
                raw_parameter_sets=[{"window": 20}],
                parsed_parameter_sets={}
            )
        ]

        # When
        parser.parse_feature_definitions(feature_defs)

        # Then
        assert mock_feature_factory.create_config_instance.call_count == 2

        # Verify all features have parsed parameters
        for feature_def in feature_defs:
            assert len(feature_def.parsed_parameter_sets) == 1
            assert "test_hash" in feature_def.parsed_parameter_sets

    def test_parse_all_parameters_empty_list(self) -> None:
        """Test parsing empty list of feature definitions."""
        # Given
        mock_feature_factory = Mock(spec=IFeatureFactory)
        parser = FeatureParameterSetParser(mock_feature_factory)
        feature_defs: list[FeatureDefinition] = []

        # When
        parser.parse_feature_definitions(feature_defs)

        # Then
        mock_feature_factory.create_config_instance.assert_not_called()

    def test_parse_all_parameters_mixed_scenarios(self) -> None:
        """Test parsing with mixed scenarios - some with params, some without."""
        # Given
        mock_feature_factory = Mock(spec=IFeatureFactory)
        config_with_params = Mock(spec=BaseParameterSetConfig)
        config_with_params.hash_id.return_value = "rsi_hash"
        mock_feature_factory.create_config_instance.return_value = config_with_params

        parser = FeatureParameterSetParser(mock_feature_factory)
        feature_defs = [
            FeatureDefinition(
                name="rsi",
                enabled=True,
                derivatives=[14],
                raw_parameter_sets=[{"period": 14}],
                parsed_parameter_sets={}
            ),
            FeatureDefinition(
                name="close_price",
                enabled=True,
                derivatives=[],
                raw_parameter_sets=[],  # No parameters
                parsed_parameter_sets={}
            )
        ]

        # When
        parser.parse_feature_definitions(feature_defs)

        # Then
        mock_feature_factory.create_config_instance.assert_called_once_with("rsi", {"period": 14})

        # RSI should have parsed parameters
        assert len(feature_defs[0].parsed_parameter_sets) == 1
        assert "rsi_hash" in feature_defs[0].parsed_parameter_sets

        # Close price should have no parsed parameters
        assert len(feature_defs[1].parsed_parameter_sets) == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_parameter_set_type_raises_error(self) -> None:
        """Test that invalid parameter set types are caught by Pydantic validation."""
        # Given & When & Then
        # Pydantic validates input types at FeatureDefinition creation time
        with pytest.raises(Exception):  # Pydantic ValidationError
            FeatureDefinition(
                name="test_feature",
                enabled=True,
                derivatives=[],
                raw_parameter_sets=["invalid_string"],  # type: ignore
                parsed_parameter_sets={}
            )

    def test_factory_exception_propagates(self) -> None:
        """Test that factory exceptions are properly propagated."""
        # Given
        mock_feature_factory = Mock(spec=IFeatureFactory)
        mock_feature_factory.create_config_instance.side_effect = RuntimeError("Factory error")
        parser = FeatureParameterSetParser(mock_feature_factory)

        feature_def = FeatureDefinition(
            name="test_feature",
            enabled=True,
            derivatives=[],
            raw_parameter_sets=[{"param": "value"}],
            parsed_parameter_sets={}
        )

        # When & Then
        with pytest.raises(RuntimeError, match="Factory error"):
            parser.parse_feature_definitions([feature_def])
