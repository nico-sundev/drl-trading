"""
Unit tests for IndicatorClassRegistry.

This module tests the core functionality of the IndicatorClassRegistry class,
including registration, discovery, and validation of indicator classes.
"""

from typing import Optional, Type
from unittest.mock import MagicMock, patch

import pytest
from drl_trading_common.base.base_indicator import BaseIndicator
from drl_trading_strategy_example.decorator.indicator_type_decorator import (
    get_indicator_type_from_class,
)
from drl_trading_strategy_example.enum.indicator_type_enum import IndicatorTypeEnum
from drl_trading_strategy_example.technical_indicator.registry.indicator_class_registry import (
    IndicatorClassRegistry,
)
from pandas import DataFrame


class InvalidIndicator:
    """Invalid indicator class that doesn't extend BaseIndicator."""
    pass


class SampleWithoutDecorator(BaseIndicator):
    """Sample indicator without decorator for testing error cases."""

    def add(self, value: DataFrame) -> None:
        pass

    def get_all(self) -> Optional[DataFrame]:
        return None

    def get_latest(self) -> Optional[DataFrame]:
        return None


class TestIndicatorClassRegistry:
    """Test cases for IndicatorClassRegistry core functionality."""

    @pytest.fixture
    def decorated_indicator_class(self) -> Type[BaseIndicator]:
        """Create a decorated indicator class for testing."""
        # Create a class with the decorator attribute manually
        class DecoratedRsiIndicator(BaseIndicator):
            _indicator_type = IndicatorTypeEnum.RSI

            def add(self, value: DataFrame) -> None:
                pass

            def get_all(self) -> Optional[DataFrame]:
                return DataFrame({"rsi": [50.0]})

            def get_latest(self) -> Optional[DataFrame]:
                return DataFrame({"rsi": [50.0]})

        return DecoratedRsiIndicator

    def test_get_indicator_class_returns_none_when_not_found(self, registry: IndicatorClassRegistry) -> None:
        """Test that get_indicator_class returns None for unregistered indicator types."""
        # Given
        indicator_type = IndicatorTypeEnum.RSI

        # When
        result = registry.get_indicator_class(indicator_type)

        # Then
        assert result is None

    def test_register_indicator_class_stores_class_successfully(
        self, registry: IndicatorClassRegistry, mock_rsi_indicator_class: Type[BaseIndicator]
    ) -> None:
        """Test that register_indicator_class stores the class and makes it retrievable."""
        # Given
        indicator_type = IndicatorTypeEnum.RSI

        # When
        registry.register_indicator_class(indicator_type, mock_rsi_indicator_class)
        result = registry.get_indicator_class(indicator_type)

        # Then
        assert result == mock_rsi_indicator_class

    def test_register_indicator_class_validates_base_indicator_interface(self, registry: IndicatorClassRegistry) -> None:
        """Test that register_indicator_class validates that classes extend BaseIndicator."""
        # Given
        indicator_type = IndicatorTypeEnum.RSI
        invalid_class = InvalidIndicator  # type: ignore[arg-type]

        # When & Then
        with pytest.raises(TypeError, match="Indicator class InvalidIndicator must extend BaseIndicator"):
            registry.register_indicator_class(indicator_type, invalid_class)

    def test_register_indicator_class_overwrites_existing_registration(
        self, registry: IndicatorClassRegistry, mock_rsi_indicator_class: Type[BaseIndicator], mock_alternative_rsi_indicator_class: Type[BaseIndicator]
    ) -> None:
        """Test that registering a class for an existing key overwrites the previous registration."""
        # Given
        indicator_type = IndicatorTypeEnum.RSI

        # When
        registry.register_indicator_class(indicator_type, mock_rsi_indicator_class)
        registry.register_indicator_class(indicator_type, mock_alternative_rsi_indicator_class)
        result = registry.get_indicator_class(indicator_type)

        # Then
        assert result == mock_alternative_rsi_indicator_class

    @patch('drl_trading_strategy_example.technical_indicator.registry.indicator_class_registry.IndicatorClassRegistry.discover_classes')
    def test_discover_indicator_classes_delegates_to_base_discovery(
        self, mock_discover: MagicMock, registry: IndicatorClassRegistry, mock_rsi_indicator_class: Type[BaseIndicator]
    ) -> None:
        """Test that discover_indicator_classes properly delegates to base class discovery method."""
        # Given
        package_name = "test_package"
        expected_result = {IndicatorTypeEnum.RSI: mock_rsi_indicator_class}
        mock_discover.return_value = expected_result

        # When
        result = registry.discover_indicator_classes(package_name)

        # Then
        mock_discover.assert_called_once_with(package_name)
        assert result == expected_result

    def test_validate_class_accepts_valid_indicator_class(
        self, registry: IndicatorClassRegistry, mock_rsi_indicator_class: Type[BaseIndicator]
    ) -> None:
        """Test that _validate_class accepts classes that properly extend BaseIndicator."""
        # Given
        # mock_rsi_indicator_class extends BaseIndicator

        # When & Then (should not raise)
        registry._validate_class(mock_rsi_indicator_class)

    def test_validate_class_rejects_invalid_class(self, registry: IndicatorClassRegistry) -> None:
        """Test that _validate_class rejects classes that don't extend BaseIndicator."""
        # Given
        invalid_class = InvalidIndicator  # type: ignore[arg-type]

        # When & Then
        with pytest.raises(TypeError, match="Indicator class InvalidIndicator must extend BaseIndicator"):
            registry._validate_class(invalid_class)

    def test_should_discover_class_returns_true_for_valid_indicator_subclass(
        self, registry: IndicatorClassRegistry, mock_rsi_indicator_class: Type[BaseIndicator]
    ) -> None:
        """Test that _should_discover_class returns True for valid BaseIndicator subclasses."""
        # Given
        # mock_rsi_indicator_class is a valid subclass of BaseIndicator

        # When
        result = registry._should_discover_class(mock_rsi_indicator_class)

        # Then
        assert result is True

    def test_should_discover_class_returns_false_for_base_indicator_itself(self, registry: IndicatorClassRegistry) -> None:
        """Test that _should_discover_class returns False for BaseIndicator class itself."""
        # Given
        base_class = BaseIndicator

        # When
        result = registry._should_discover_class(base_class)

        # Then
        assert result is False

    def test_should_discover_class_returns_false_for_non_indicator_class(self, registry: IndicatorClassRegistry) -> None:
        """Test that _should_discover_class returns False for classes that don't extend BaseIndicator."""
        # Given
        non_indicator_class = InvalidIndicator

        # When
        result = registry._should_discover_class(non_indicator_class)

        # Then
        assert result is False

    @patch('drl_trading_strategy_example.technical_indicator.registry.indicator_class_registry.get_indicator_type_from_class')
    def test_extract_key_from_class_uses_decorator_function(
        self, mock_get_indicator_type: MagicMock, registry: IndicatorClassRegistry,
        decorated_indicator_class: Type[BaseIndicator]
    ) -> None:
        """Test that _extract_key_from_class delegates to get_indicator_type_from_class function."""
        # Given
        expected_indicator_type = IndicatorTypeEnum.RSI
        mock_get_indicator_type.return_value = expected_indicator_type

        # When
        result = registry._extract_key_from_class(decorated_indicator_class)

        # Then
        mock_get_indicator_type.assert_called_once_with(decorated_indicator_class)
        assert result == expected_indicator_type

    def test_extract_key_from_class_with_decorated_class(
        self, registry: IndicatorClassRegistry, decorated_indicator_class: Type[BaseIndicator]
    ) -> None:
        """Test that _extract_key_from_class works with properly decorated classes."""
        # Given
        # decorated_indicator_class has _indicator_type attribute set

        # When
        result = registry._extract_key_from_class(decorated_indicator_class)

        # Then
        assert result == IndicatorTypeEnum.RSI

    def test_extract_key_from_class_raises_error_for_undecorated_class(
        self, registry: IndicatorClassRegistry
    ) -> None:
        """Test that _extract_key_from_class raises AttributeError for classes without decorator."""
        # Given
        undecorated_class = SampleWithoutDecorator

        # When & Then
        with pytest.raises(AttributeError, match="Class SampleWithoutDecorator has no indicator type information"):
            registry._extract_key_from_class(undecorated_class)

    def test_registry_inherits_from_discoverable_registry(self, registry: IndicatorClassRegistry) -> None:
        """Test that IndicatorClassRegistry properly inherits from DiscoverableRegistry."""
        # Given
        # registry instance

        # When & Then
        assert hasattr(registry, 'get_class')
        assert hasattr(registry, 'register_class')
        assert hasattr(registry, 'discover_classes')
        assert hasattr(registry, 'reset')

    def test_registry_type_parameters_are_correct(self, registry: IndicatorClassRegistry, mock_rsi_indicator_class: Type[BaseIndicator]) -> None:
        """Test that the registry uses correct type parameters for IndicatorTypeEnum and BaseIndicator."""
        # Given
        indicator_type = IndicatorTypeEnum.RSI

        # When
        registry.register_indicator_class(indicator_type, mock_rsi_indicator_class)
        result = registry.get_indicator_class(indicator_type)

        # Then
        assert result == mock_rsi_indicator_class
        assert isinstance(indicator_type, IndicatorTypeEnum)
        assert issubclass(mock_rsi_indicator_class, BaseIndicator)

    def test_registry_handles_empty_state_gracefully(self, registry: IndicatorClassRegistry) -> None:
        """Test that registry methods handle empty state without errors."""
        # Given
        empty_registry = registry

        # When
        result = empty_registry.get_indicator_class(IndicatorTypeEnum.RSI)

        # Then
        assert result is None

    def test_integration_with_get_indicator_type_from_class_function(
        self, registry: IndicatorClassRegistry, decorated_indicator_class: Type[BaseIndicator]
    ) -> None:
        """Test integration with the get_indicator_type_from_class utility function."""
        # Given
        # decorated_indicator_class has proper decorator attribute

        # When
        extracted_type = get_indicator_type_from_class(decorated_indicator_class)
        registry_extracted_type = registry._extract_key_from_class(decorated_indicator_class)

        # Then
        assert extracted_type == IndicatorTypeEnum.RSI
        assert registry_extracted_type == extracted_type
