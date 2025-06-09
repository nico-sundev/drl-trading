"""
Unit tests for FeatureConfigRegistry.

Tests the registry's ability to discover, register, and manage feature configuration class types
using the modern @feature_type decorator approach.
"""

from typing import Type
from unittest.mock import MagicMock, patch

import pytest
from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_strategy.decorator.feature_type_decorator import feature_type
from drl_trading_strategy.enum.feature_type_enum import FeatureTypeEnum
from drl_trading_strategy.feature.registry.feature_config_registry import (
    FeatureConfigRegistry,
)


@feature_type(FeatureTypeEnum.RSI)
class SampleRsiConfig(BaseParameterSetConfig):
    """Sample RSI config class using @feature_type decorator."""

    type: str = "rsi"
    enabled: bool = True
    length: int = 14

class ConfigWithoutDecorator(BaseParameterSetConfig):
    """Config class without @feature_type decorator (should cause errors)."""

    type: str = "undecorated"
    enabled: bool = True
    value: int = 42


@feature_type(FeatureTypeEnum.RSI)
class SampleConfigWithDecorator(BaseParameterSetConfig):
    """Test config class with @feature_type decorator."""

    type: str = "rsi"
    enabled: bool = True
    length: int = 14


class InvalidConfigClass:
    """Test class that does not extend BaseParameterSetConfig."""

    def some_method(self) -> str:
        return "invalid"


class TestFeatureConfigRegistry:
    """Test cases for FeatureConfigRegistry core functionality."""

    @pytest.fixture
    def registry(self) -> FeatureConfigRegistry:
        """Provide a fresh registry instance for each test."""
        return FeatureConfigRegistry()

    @pytest.fixture
    def decorated_config_class(self) -> Type[BaseParameterSetConfig]:
        """Provide a config class with @feature_type decorator."""
        @feature_type(FeatureTypeEnum.RSI)
        class DecoratedConfig(BaseParameterSetConfig):
            type: str = "rsi"
            enabled: bool = True
            length: int = 14
        return DecoratedConfig

    def test_get_config_class_returns_registered_class(self, registry: FeatureConfigRegistry, decorated_config_class: Type[BaseParameterSetConfig]) -> None:
        """Test that get_config_class returns a previously registered config class."""
        # Given
        feature_type_string = "rsi"
        registry.register_config_class(feature_type_string, decorated_config_class)
          # When
        result = registry.get_config_class(feature_type_string)

        # Then
        assert result is decorated_config_class

    def test_get_config_class_returns_none_for_unregistered_type(self, registry: FeatureConfigRegistry) -> None:
        """Test that get_config_class raises ValueError for unregistered feature types."""
        # Given
        unregistered_feature_type = "nonexistent"

        # When/Then
        with pytest.raises(ValueError, match="is not a valid FeatureTypeEnum value"):
            registry.get_config_class(unregistered_feature_type)

    def test_register_config_class_stores_class_successfully(self, registry: FeatureConfigRegistry, decorated_config_class: Type[BaseParameterSetConfig]) -> None:
        """Test that register_config_class properly stores a config class."""
        # Given
        feature_type_string = "rsi"

        # When
        registry.register_config_class(feature_type_string, decorated_config_class)
        result = registry.get_config_class(feature_type_string)

        # Then
        assert result is decorated_config_class

    def test_register_config_class_handles_case_insensitive_lookup(self, registry: FeatureConfigRegistry, decorated_config_class: Type[BaseParameterSetConfig]) -> None:
        """Test that config class registration and lookup are case-insensitive."""
        # Given
        registry.register_config_class("RSI", decorated_config_class)

        # When
        result_lower = registry.get_config_class("rsi")
        result_upper = registry.get_config_class("RSI")
        result_mixed = registry.get_config_class("Rsi")

        # Then
        assert result_lower is decorated_config_class
        assert result_upper is decorated_config_class
        assert result_mixed is decorated_config_class

    @patch('drl_trading_strategy.feature.registry.feature_config_registry.FeatureConfigRegistry.discover_classes')
    def test_discover_config_classes_delegates_to_base_class(self, mock_discover_classes: MagicMock, registry: FeatureConfigRegistry) -> None:
        """Test that discover_config_classes delegates to the base class discover_classes method."""
        # Given
        package_name = "test.package"
        expected_result = {FeatureTypeEnum.RSI: SampleRsiConfig}
        mock_discover_classes.return_value = expected_result

        # When
        result = registry.discover_config_classes(package_name)

        # Then
        mock_discover_classes.assert_called_once_with(package_name)
        assert result == expected_result


class TestFeatureConfigRegistryValidation:
    """Test cases for FeatureConfigRegistry validation logic."""
    @pytest.fixture
    def registry(self) -> FeatureConfigRegistry:
        """Provide a fresh registry instance for each test."""
        return FeatureConfigRegistry()

    def test_validate_class_accepts_valid_config_class(self, registry: FeatureConfigRegistry) -> None:
        """Test that _validate_class accepts classes that extend BaseParameterSetConfig."""
        # Given
        valid_config_class = SampleRsiConfig

        # When/Then
        # Should not raise any exception
        registry._validate_class(valid_config_class)

    def test_validate_class_rejects_invalid_class(self, registry: FeatureConfigRegistry) -> None:
        """Test that _validate_class rejects classes that don't extend BaseParameterSetConfig."""
        # Given
        invalid_class = InvalidConfigClass
          # When/Then
        with pytest.raises(TypeError, match="must extend BaseParameterSetConfig"):
            registry._validate_class(invalid_class)

    def test_should_discover_class_returns_true_for_valid_config(self, registry: FeatureConfigRegistry) -> None:
        """Test that _should_discover_class returns True for valid config classes."""
        # Given
        valid_config_class = SampleRsiConfig

        # When
        result = registry._should_discover_class(valid_config_class)

        # Then
        assert result is True

    def test_should_discover_class_returns_false_for_base_class(self, registry: FeatureConfigRegistry) -> None:
        """Test that _should_discover_class returns False for BaseParameterSetConfig itself."""
        # Given
        base_class = BaseParameterSetConfig

        # When
        result = registry._should_discover_class(base_class)

        # Then
        assert result is False

    def test_should_discover_class_returns_false_for_invalid_class(self, registry: FeatureConfigRegistry) -> None:
        """Test that _should_discover_class returns False for classes that don't extend BaseParameterSetConfig."""
        # Given
        invalid_class = InvalidConfigClass

        # When
        result = registry._should_discover_class(invalid_class)

        # Then
        assert result is False


class TestFeatureConfigRegistryKeyExtraction:
    """Test cases for FeatureConfigRegistry key extraction logic."""
    @pytest.fixture
    def registry(self) -> FeatureConfigRegistry:
        """Provide a fresh registry instance for each test."""
        return FeatureConfigRegistry()

    @patch('drl_trading_strategy.feature.registry.feature_config_registry.get_feature_type_from_class')
    def test_extract_key_from_class_uses_decorator_utility(self, mock_get_feature_type: MagicMock, registry: FeatureConfigRegistry) -> None:
        """Test that _extract_key_from_class uses the decorator utility function."""
        # Given
        test_class = SampleRsiConfig
        expected_enum = FeatureTypeEnum.RSI
        mock_get_feature_type.return_value = expected_enum

        # When
        result = registry._extract_key_from_class(test_class)

        # Then
        mock_get_feature_type.assert_called_once_with(test_class)
        assert result == expected_enum

    @patch('drl_trading_strategy.feature.registry.feature_config_registry.get_feature_type_from_class')
    @patch('drl_trading_strategy.feature.registry.feature_config_registry.FeatureTypeConverter.string_to_enum')
    def test_extract_key_removes_config_suffix_in_fallback(self, mock_string_to_enum: MagicMock, mock_get_feature_type: MagicMock, registry: FeatureConfigRegistry) -> None:
        """Test that _extract_key_from_class removes 'Config' suffix when falling back to name-based conversion."""
        # Given
        class RsiConfig(BaseParameterSetConfig):
            type: str = "rsi"
            enabled: bool = True

        mock_get_feature_type.side_effect = AttributeError("No feature type information")
        expected_enum = FeatureTypeEnum.RSI
        mock_string_to_enum.return_value = expected_enum

        # When
        result = registry._extract_key_from_class(RsiConfig)

        # Then
        mock_string_to_enum.assert_called_once_with("rsi")
        assert result == expected_enum


class TestFeatureConfigRegistryIntegration:
    """Integration tests for FeatureConfigRegistry with real dependencies."""

    @pytest.fixture
    def registry(self) -> FeatureConfigRegistry:
        """Provide a fresh registry instance for each test."""
        return FeatureConfigRegistry()

    def test_full_workflow_with_decorated_config_class(self, registry: FeatureConfigRegistry) -> None:
        """Test complete workflow using a config class with @feature_type decorator."""
        # Given
        @feature_type(FeatureTypeEnum.RSI)
        class RsiConfig(BaseParameterSetConfig):
            type: str = "rsi"
            enabled: bool = True
            length: int = 14

        # When
        # Register the config class
        registry.register_config_class("rsi", RsiConfig)

        # Retrieve the config class
        result = registry.get_config_class("rsi")

        # Then
        assert result is RsiConfig
        assert issubclass(result, BaseParameterSetConfig)

    def test_validation_integration_with_real_classes(self, registry: FeatureConfigRegistry) -> None:
        """Test validation logic with real config classes."""
        # Given
        valid_class = SampleConfigWithDecorator
        invalid_class = InvalidConfigClass

        # When/Then
        # Valid class should pass validation
        assert registry._should_discover_class(valid_class) is True
        registry._validate_class(valid_class)  # Should not raise

        # Invalid class should fail validation
        assert registry._should_discover_class(invalid_class) is False
        with pytest.raises(TypeError):
            registry._validate_class(invalid_class)

    def test_key_extraction_with_decorated_class(self, registry: FeatureConfigRegistry) -> None:
        """Test key extraction with a real decorated config class."""
        # Given
        @feature_type(FeatureTypeEnum.RSI)
        class DecoratedRsiConfig(BaseParameterSetConfig):
            type: str = "rsi"
            enabled: bool = True
            length: int = 14

        # When
        result = registry._extract_key_from_class(DecoratedRsiConfig)

        # Then
        assert result == FeatureTypeEnum.RSI


class TestFeatureConfigRegistryErrorHandling:
    """Test cases for FeatureConfigRegistry error conditions and edge cases."""

    @pytest.fixture
    def registry(self) -> FeatureConfigRegistry:
        """Provide a fresh registry instance for each test."""
        return FeatureConfigRegistry()

    @patch('drl_trading_strategy.feature.registry.feature_config_registry.FeatureTypeConverter.string_to_enum')
    def test_get_config_class_handles_converter_exceptions(self, mock_string_to_enum: MagicMock, registry: FeatureConfigRegistry) -> None:
        """Test that get_config_class handles FeatureTypeConverter exceptions gracefully."""
        # Given
        mock_string_to_enum.side_effect = ValueError("Invalid feature type")

        # When/Then
        with pytest.raises(ValueError, match="Invalid feature type"):
            registry.get_config_class("invalid_type")

    @patch('drl_trading_strategy.feature.registry.feature_config_registry.FeatureTypeConverter.string_to_enum')
    def test_register_config_class_handles_converter_exceptions(self, mock_string_to_enum: MagicMock, registry: FeatureConfigRegistry) -> None:
        """Test that register_config_class handles FeatureTypeConverter exceptions gracefully."""
        # Given
        mock_string_to_enum.side_effect = ValueError("Invalid feature type")
        test_class = SampleConfigWithDecorator

        # When/Then
        with pytest.raises(ValueError, match="Invalid feature type"):
            registry.register_config_class("invalid_type", test_class)

    def test_register_config_class_with_none_class_raises_error(self, registry: FeatureConfigRegistry) -> None:
        """Test that register_config_class handles None config class appropriately."""
        # Given
        feature_type = "rsi"

        # When/Then
        # This should cause a validation error when the base registry tries to validate
        with pytest.raises((TypeError, AttributeError)):
            registry.register_config_class(feature_type, None)  # type: ignore

    def test_extract_key_from_class_with_none_class(self, registry: FeatureConfigRegistry) -> None:
        """Test that _extract_key_from_class handles None class input gracefully."""
        # Given/When/Then
        with pytest.raises(AttributeError):
            registry._extract_key_from_class(None)  # type: ignore

    @patch('drl_trading_strategy.feature.registry.feature_config_registry.get_feature_type_from_class')
    @patch('drl_trading_strategy.feature.registry.feature_config_registry.FeatureTypeConverter.string_to_enum')
    def test_extract_key_handles_both_decorator_and_converter_failures(self, mock_string_to_enum: MagicMock, mock_get_feature_type: MagicMock, registry: FeatureConfigRegistry) -> None:
        """Test that _extract_key_from_class handles failures in both decorator and converter fallback."""
        # Given
        test_class = SampleConfigWithDecorator
        mock_get_feature_type.side_effect = AttributeError("No feature type")
        mock_string_to_enum.side_effect = ValueError("Invalid conversion")

        # When/Then
        with pytest.raises(ValueError, match="Invalid conversion"):
            registry._extract_key_from_class(test_class)


class TestFeatureConfigRegistryInheritance:
    """Test cases for FeatureConfigRegistry inheritance from DiscoverableRegistry."""

    @pytest.fixture
    def registry(self) -> FeatureConfigRegistry:
        """Provide a fresh registry instance for each test."""
        return FeatureConfigRegistry()

    def test_inherits_from_discoverable_registry(self, registry: FeatureConfigRegistry) -> None:
        """Test that FeatureConfigRegistry properly inherits from DiscoverableRegistry."""
        # Given
        from drl_trading_common.base.discoverable_registry import DiscoverableRegistry

        # When/Then
        assert isinstance(registry, DiscoverableRegistry)

    def test_implements_feature_config_registry_interface(self, registry: FeatureConfigRegistry) -> None:
        """Test that FeatureConfigRegistry implements FeatureConfigRegistryInterface."""
        # Given
        from drl_trading_common.interfaces.feature.registry.feature_config_registry_interface import (
            FeatureConfigRegistryInterface,
        )

        # When/Then
        assert isinstance(registry, FeatureConfigRegistryInterface)

    def test_has_required_interface_methods(self, registry: FeatureConfigRegistry) -> None:
        """Test that FeatureConfigRegistry has all required interface methods."""
        # Given/When/Then
        assert hasattr(registry, 'get_config_class')
        assert hasattr(registry, 'register_config_class')
        assert hasattr(registry, 'reset')  # From base interface
        assert callable(registry.get_config_class)
        assert callable(registry.register_config_class)
        assert callable(registry.reset)
