"""
Unit tests for FeatureClassRegistry.

Tests the registry's ability to discover, register, and manage feature class types
using the modern @feature_type decorator approach.
"""

from typing import Optional, Type
from unittest.mock import MagicMock, patch

import pytest
from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_common.interfaces.indicator.technical_indicator_facade_interface import (
    TechnicalIndicatorFacadeInterface,
)
from drl_trading_strategy.decorator.feature_type_decorator import feature_type
from drl_trading_strategy.enum.feature_type_enum import FeatureTypeEnum
from drl_trading_strategy.feature.registry.feature_class_registry import (
    FeatureClassRegistry,
)
from pandas import DataFrame


# Mock technical indicator interface to avoid import issues
class MockTechnicalIndicatorFacade(TechnicalIndicatorFacadeInterface):
    """Mock technical indicator facade for testing."""
    pass


@feature_type(FeatureTypeEnum.RSI)
class SampleRsiFeature(BaseFeature):
    """Sample RSI feature class with @feature_type decorator."""

    def __init__(self, source: DataFrame, config: BaseParameterSetConfig, indicator_service: MockTechnicalIndicatorFacade, postfix: str = "") -> None:
        super().__init__(source, config, indicator_service, postfix)

    def add(self, df: DataFrame) -> None:
        pass

    def compute_latest(self) -> Optional[DataFrame]:
        return None

    def compute_all(self) -> Optional[DataFrame]:
        return None

    def get_sub_features_names(self) -> list[str]:
        return ["rsi"]


class RsiFeature(BaseFeature):
    """Feature class without @feature_type decorator (should cause errors)."""

    def __init__(self, source: DataFrame, config: BaseParameterSetConfig, indicator_service: MockTechnicalIndicatorFacade, postfix: str = "") -> None:
        super().__init__(source, config, indicator_service, postfix)

    def add(self, df: DataFrame) -> None:
        pass

    def compute_latest(self) -> Optional[DataFrame]:
        return None

    def compute_all(self) -> Optional[DataFrame]:
        return None

    def get_sub_features_names(self) -> list[str]:
        return ["undecorated"]


class InvalidFeatureClass:
    """Test class that does not extend BaseFeature."""

    def some_method(self) -> str:
        return "invalid"


class TestFeatureClassRegistry:
    """Test cases for FeatureClassRegistry core functionality."""

    @pytest.fixture
    def registry(self) -> FeatureClassRegistry:
        """Create a fresh FeatureClassRegistry instance for each test."""
        return FeatureClassRegistry()

    @pytest.fixture
    def decorated_feature_class(self) -> Type[BaseFeature]:
        """Provide a feature class with @feature_type decorator."""
        return SampleRsiFeature

    def test_get_feature_class_returns_registered_class(self, registry: FeatureClassRegistry, decorated_feature_class: Type[BaseFeature]) -> None:
        """Test that get_feature_class returns a previously registered feature class."""
        # Given
        feature_type_string = "rsi"
        registry.register_feature_class(feature_type_string, decorated_feature_class)

        # When
        result = registry.get_feature_class(feature_type_string)

        # Then
        assert result == decorated_feature_class

    def test_register_feature_class_stores_class_successfully(self, registry: FeatureClassRegistry, decorated_feature_class: Type[BaseFeature]) -> None:
        """Test that register_feature_class successfully stores a feature class."""
        # Given
        feature_type_string = "rsi"

        # When
        registry.register_feature_class(feature_type_string, decorated_feature_class)

        # Then
        stored_class = registry.get_feature_class(feature_type_string)
        assert stored_class == decorated_feature_class

    def test_register_feature_class_handles_case_insensitive_lookup(self, registry: FeatureClassRegistry, decorated_feature_class: Type[BaseFeature]) -> None:
        """Test that register_feature_class works with case-insensitive feature type strings."""
        # Given
        feature_type_string = "RSI"

        # When
        registry.register_feature_class(feature_type_string, decorated_feature_class)

        # Then
        stored_class = registry.get_feature_class("rsi")
        assert stored_class == decorated_feature_class

    @patch('drl_trading_strategy.feature.registry.feature_class_registry.FeatureClassRegistry.discover_classes')
    def test_discover_feature_classes_delegates_to_base_discovery(self, mock_discover: MagicMock, registry: FeatureClassRegistry) -> None:
        """Test that discover_feature_classes properly delegates to base class discovery method."""
        # Given
        package_name = "test_package"
        expected_result = {FeatureTypeEnum.RSI: SampleRsiFeature}
        mock_discover.return_value = expected_result

        # When
        result = registry.discover_feature_classes(package_name)

        # Then
        mock_discover.assert_called_once_with(package_name)
        assert result == expected_result


class TestFeatureClassRegistryValidation:
    """Test cases for FeatureClassRegistry validation logic."""

    @pytest.fixture
    def registry(self) -> FeatureClassRegistry:
        """Create a fresh FeatureClassRegistry instance for each test."""
        return FeatureClassRegistry()

    def test_validate_class_accepts_valid_feature_class(self, registry: FeatureClassRegistry) -> None:
        """Test that _validate_class accepts classes that extend BaseFeature."""
        # Given
        valid_feature_class = SampleRsiFeature

        # When/Then
        # Should not raise any exception
        registry._validate_class(valid_feature_class)

    def test_validate_class_rejects_invalid_class(self, registry: FeatureClassRegistry) -> None:
        """Test that _validate_class rejects classes that don't extend BaseFeature."""
        # Given
        invalid_class = InvalidFeatureClass

        # When/Then
        with pytest.raises(TypeError, match="Feature class InvalidFeatureClass must extend BaseFeature"):
            registry._validate_class(invalid_class)

    def test_should_discover_class_returns_true_for_valid_feature(self, registry: FeatureClassRegistry) -> None:
        """Test that _should_discover_class returns True for valid feature classes."""
        # Given
        valid_feature_class = SampleRsiFeature

        # When
        result = registry._should_discover_class(valid_feature_class)

        # Then
        assert result is True

    def test_should_discover_class_returns_false_for_base_feature(self, registry: FeatureClassRegistry) -> None:
        """Test that _should_discover_class returns False for BaseFeature itself."""
        # Given
        base_feature_class = BaseFeature

        # When
        result = registry._should_discover_class(base_feature_class)

        # Then
        assert result is False

    def test_should_discover_class_returns_false_for_invalid_class(self, registry: FeatureClassRegistry) -> None:
        """Test that _should_discover_class returns False for classes not extending BaseFeature."""
        # Given
        invalid_class = InvalidFeatureClass

        # When
        result = registry._should_discover_class(invalid_class)

        # Then
        assert result is False


class TestFeatureClassRegistryKeyExtraction:
    """Test cases for FeatureClassRegistry key extraction logic."""

    @pytest.fixture
    def registry(self) -> FeatureClassRegistry:
        """Create a fresh FeatureClassRegistry instance for each test."""
        return FeatureClassRegistry()

    def test_extract_key_from_class_uses_decorator(self, registry: FeatureClassRegistry) -> None:
        """Test that _extract_key_from_class uses @feature_type decorator when available."""
        # Given
        decorated_class = SampleRsiFeature

        # When
        result = registry._extract_key_from_class(decorated_class)

        # Then
        assert result == FeatureTypeEnum.RSI

    def test_extract_key_from_class_name_based_fallback(self, registry: FeatureClassRegistry) -> None:
        """Test that _extract_key_from_class can extract feature type from class name as fallback."""
        # Given
        undecorated_class = RsiFeature

        # When
        result = registry._extract_key_from_class(undecorated_class)

        # Then
        # Should extract "rsi" from class name, which maps to a valid enum
        assert isinstance(result, FeatureTypeEnum)


class TestFeatureClassRegistryIntegration:
    """Integration test cases for FeatureClassRegistry."""

    @pytest.fixture
    def registry(self) -> FeatureClassRegistry:
        """Create a fresh FeatureClassRegistry instance for each test."""
        return FeatureClassRegistry()

    def test_full_registration_and_retrieval_workflow(self, registry: FeatureClassRegistry) -> None:
        """Test complete workflow of registering and retrieving feature classes."""
        # Given
        feature_classes = {
            "rsi": SampleRsiFeature,
        }

        # When
        for feature_type_name, feature_class in feature_classes.items():
            registry.register_feature_class(feature_type_name, feature_class)

        # Then
        for feature_type_name, expected_class in feature_classes.items():
            retrieved_class = registry.get_feature_class(feature_type_name)
            assert retrieved_class == expected_class

    def test_decorator_based_key_extraction_integration(self, registry: FeatureClassRegistry) -> None:
        """Test that feature classes with decorators can be properly processed."""
        # Given
        decorated_class = SampleRsiFeature

        # When
        extracted_key = registry._extract_key_from_class(decorated_class)
        registry.register_class(extracted_key, decorated_class)

        # Then
        retrieved_class = registry.get_class(FeatureTypeEnum.RSI)
        assert retrieved_class == decorated_class

class TestFeatureClassRegistryErrorHandling:
    """Test cases for FeatureClassRegistry error handling scenarios."""

    @pytest.fixture
    def registry(self) -> FeatureClassRegistry:
        """Create a fresh FeatureClassRegistry instance for each test."""
        return FeatureClassRegistry()

    def test_register_invalid_feature_class_raises_error(self, registry: FeatureClassRegistry) -> None:
        """Test that registering an invalid feature class raises appropriate error."""
        # Given
        invalid_class = InvalidFeatureClass
        feature_type_name = "invalid"

        # When/Then
        with pytest.raises(ValueError):
            registry.register_feature_class(feature_type_name, invalid_class)

    def test_discover_classes_with_empty_package(self, registry: FeatureClassRegistry) -> None:
        """Test that discover_feature_classes handles empty package names gracefully."""
        # Given
        empty_package = ""

        # When & Then
        with pytest.raises(ValueError):
            registry.discover_feature_classes(empty_package)


class TestFeatureClassRegistryInheritance:
    """Test cases for FeatureClassRegistry inheritance and polymorphism scenarios."""

    @pytest.fixture
    def registry(self) -> FeatureClassRegistry:
        """Create a fresh FeatureClassRegistry instance for each test."""
        return FeatureClassRegistry()

    def test_multiple_inheritance_feature_registration(self, registry: FeatureClassRegistry) -> None:
        """Test that feature classes with multiple inheritance work correctly."""
        # Given
        class HelperMixin:
            def helper_method(self) -> str:
                return "helper"

        @feature_type(FeatureTypeEnum.RSI)
        class MixinFeature(SampleRsiFeature, HelperMixin):
            """Feature class with multiple inheritance."""
            pass

        feature_type_name = "rsi"

        # When
        registry.register_feature_class(feature_type_name, MixinFeature)

        # Then
        retrieved_class = registry.get_feature_class(feature_type_name)
        assert retrieved_class == MixinFeature
        assert issubclass(retrieved_class, BaseFeature)
        assert issubclass(retrieved_class, HelperMixin)

    def test_abstract_feature_class_handling(self, registry: FeatureClassRegistry) -> None:
        """Test that abstract feature classes are handled appropriately."""
        # Given
        from abc import ABC, abstractmethod

        class AbstractFeature(BaseFeature, ABC):
            """Abstract feature class."""

            @abstractmethod
            def abstract_method(self) -> str:
                pass

        # When
        should_discover = registry._should_discover_class(AbstractFeature)

        # Then
        # Abstract classes should still be discoverable if they extend BaseFeature
        # but are not BaseFeature itself
        assert should_discover is True
