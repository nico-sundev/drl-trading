"""
Unit tests for feature_type_decorator module.

Tests the @feature_type decorator and get_feature_type_from_class utility function
to ensure proper feature type registration and extraction.
"""

from unittest.mock import Mock

import pytest
from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_strategy_example.decorator.feature_type_decorator import (
    feature_type,
    get_feature_type_from_class,
)
from drl_trading_strategy_example.enum.feature_type_enum import FeatureTypeEnum
from pandas import DataFrame


class TestFeatureTypeDecorator:
    """Test cases for the @feature_type decorator."""

    @pytest.fixture
    def sample_feature_enum(self) -> FeatureTypeEnum:
        """Provide a sample FeatureTypeEnum for testing."""
        return FeatureTypeEnum.RSI

    def test_decorator_sets_feature_type_attribute(self, sample_feature_enum: FeatureTypeEnum) -> None:
        """Test that @feature_type decorator sets _feature_type class attribute."""
        # Given
        # A class that will be decorated with a specific feature type

        # When
        @feature_type(sample_feature_enum)
        class TestFeature(BaseFeature):
            def compute(self) -> DataFrame:
                return DataFrame()

            def get_sub_features_names(self) -> list[str]:
                return ["test"]

            def get_feature_name(self) -> str:
                return "test"

            def get_config_to_string(self) -> str:
                return "A1b2c3"

        # Then
        assert hasattr(TestFeature, '_feature_type')
        assert TestFeature._feature_type == sample_feature_enum
        assert isinstance(TestFeature._feature_type, FeatureTypeEnum)

    def test_decorator_does_not_add_static_method(self, sample_feature_enum: FeatureTypeEnum) -> None:
        """Test that @feature_type decorator only sets _feature_type attribute (no static method)."""
        # Given
        # A class that will be decorated with a specific feature type

        # When
        @feature_type(sample_feature_enum)
        class TestFeature(BaseFeature):
            def compute(self) -> DataFrame:
                return DataFrame()

            def get_sub_features_names(self) -> list[str]:
                return ["test"]

            def get_feature_name(self) -> str:
                return "test"

            def get_config_to_string(self) -> str:
                return "A1b2c3"

        # Then
        assert hasattr(TestFeature, '_feature_type')
        assert TestFeature._feature_type == sample_feature_enum
        # Verify that the decorator does NOT add a get_feature_type method
        assert not hasattr(TestFeature, 'get_feature_type')

    def test_decorator_preserves_original_class_functionality(self, sample_feature_enum: FeatureTypeEnum) -> None:
        """Test that @feature_type decorator preserves original class methods and attributes."""
        # Given
        # A class with specific methods and attributes that will be decorated

        # When
        @feature_type(sample_feature_enum)
        class TestFeature(BaseFeature):
            custom_attribute = "test_value"

            def compute(self) -> DataFrame:
                return DataFrame({"test": [1, 2, 3]})

            def compute_all(self) -> DataFrame:
                return self.compute()

            def add(self, df: DataFrame) -> None:
                pass

            def compute_latest(self) -> DataFrame:
                return self.compute()

            def get_sub_features_names(self) -> list[str]:
                return ["test_feature"]

            def get_feature_name(self) -> str:
                return "test_feature"

            def custom_method(self) -> str:
                return "custom_result"

            def get_config_to_string(self) -> str:
                return "A1b2c3"

        # Then
        instance = TestFeature(Mock(), Mock(), Mock())
        assert instance.custom_attribute == "test_value"
        assert instance.custom_method() == "custom_result"
        assert instance.get_feature_name() == "test_feature"
        assert instance.get_sub_features_names() == ["test_feature"]
        result_df = instance.compute()
        assert not result_df.empty
        assert "test" in result_df.columns

    def test_decorator_works_with_rsi_feature_type(self) -> None:
        """Test that @feature_type decorator works with RSI FeatureTypeEnum value."""
        # Given
        # RSI feature type to test

        # When
        @feature_type(FeatureTypeEnum.RSI)
        class RsiFeature(BaseFeature):
            def compute(self) -> DataFrame:
                return DataFrame()

            def compute_all(self) -> DataFrame:
                return self.compute()

            def add(self, df: DataFrame) -> None:
                pass

            def compute_latest(self) -> DataFrame:
                return self.compute()

            def get_sub_features_names(self) -> list[str]:
                return ["rsi"]

            def get_feature_name(self) -> str:
                return "rsi"

            def get_config_to_string(self) -> str:
                return "A1b2c3"

        # Then
        assert RsiFeature._feature_type == FeatureTypeEnum.RSI

    def test_decorator_works_with_inheritance(self, sample_feature_enum: FeatureTypeEnum) -> None:
        """Test that @feature_type decorator works correctly with class inheritance."""
        # Given
        # A base class and a derived class, both decorated

        # When
        @feature_type(sample_feature_enum)
        class BaseTestFeature(BaseFeature):
            def compute(self) -> DataFrame:
                return DataFrame()

            def get_sub_features_names(self) -> list[str]:
                return ["base"]

            def get_feature_name(self) -> str:
                return "base"

            def get_config_to_string(self) -> str:
                return "A1b2c3"

        @feature_type(FeatureTypeEnum.RSI)
        class DerivedTestFeature(BaseTestFeature):
            def get_feature_name(self) -> str:
                return "derived"

        # Then
        assert BaseTestFeature._feature_type == sample_feature_enum
        assert DerivedTestFeature._feature_type == FeatureTypeEnum.RSI

    def test_decorator_with_non_basefeature_class(self, sample_feature_enum: FeatureTypeEnum) -> None:
        """Test that @feature_type decorator works with non-BaseFeature classes."""
        # Given
        # A class that doesn't inherit from BaseFeature

        # When
        @feature_type(sample_feature_enum)
        class NonFeatureClass:
            def some_method(self) -> str:
                return "test"

        # Then
        assert hasattr(NonFeatureClass, '_feature_type')
        assert NonFeatureClass._feature_type == sample_feature_enum


class TestGetFeatureTypeFromClass:
    """Test cases for the get_feature_type_from_class utility function."""

    @pytest.fixture
    def sample_feature_enum(self) -> FeatureTypeEnum:
        """Provide a sample FeatureTypeEnum for testing."""
        return FeatureTypeEnum.RSI

    def test_extract_from_decorated_class(self, sample_feature_enum: FeatureTypeEnum) -> None:
        """Test extracting feature type from @feature_type decorated class."""
        # Given
        @feature_type(sample_feature_enum)
        class DecoratedFeature(BaseFeature):
            def compute(self) -> DataFrame:
                return DataFrame()

            def get_sub_features_names(self) -> list[str]:
                return ["decorated"]

            def get_feature_name(self) -> str:
                return "decorated"

            def get_config_to_string(self) -> str:
                return "A1b2c3"

        # When
        result = get_feature_type_from_class(DecoratedFeature)

        # Then
        assert result == sample_feature_enum
        assert isinstance(result, FeatureTypeEnum)

    def test_extract_from_undecorated_class_raises_error(self) -> None:
        """Test that extracting from undecorated class raises AttributeError."""
        # Given
        class UndecoratedFeature(BaseFeature):
            def compute(self) -> DataFrame:
                return DataFrame()

            def get_sub_features_names(self) -> list[str]:
                return ["undecorated"]

            def get_feature_name(self) -> str:
                return "undecorated"

            def get_config_to_string(self) -> str:
                return "A1b2c3"

        # When & Then
        with pytest.raises(AttributeError, match="has no feature type information"):
            get_feature_type_from_class(UndecoratedFeature)

    def test_raises_attribute_error_when_no_feature_type_info(self) -> None:
        """Test that AttributeError is raised when class has no feature type information."""
        # Given
        class NoFeatureTypeClass:
            def some_method(self) -> str:
                return "test"

        # When/Then
        with pytest.raises(AttributeError) as exc_info:
            get_feature_type_from_class(NoFeatureTypeClass)

        assert "has no feature type information" in str(exc_info.value)
        assert "NoFeatureTypeClass" in str(exc_info.value)

    def test_raises_attribute_error_with_proper_message(self) -> None:
        """Test that AttributeError has descriptive message with usage instructions."""
        # Given
        class TestClass:
            pass

        # When/Then
        with pytest.raises(AttributeError) as exc_info:
            get_feature_type_from_class(TestClass)

        error_message = str(exc_info.value)
        assert "TestClass" in error_message
        assert "has no feature type information" in error_message
        assert "@feature_type decorator" in error_message

    def test_works_with_rsi_feature_type_enum(self) -> None:
        """Test that utility function works with RSI FeatureTypeEnum value."""
        # Given
        feature_enum = FeatureTypeEnum.RSI

        # When
        @feature_type(feature_enum)
        class TestFeature(BaseFeature):
            def compute(self) -> DataFrame:
                return DataFrame()

            def get_sub_features_names(self) -> list[str]:
                return ["test"]

            def get_feature_name(self) -> str:
                return "test"

            def get_config_to_string(self) -> str:
                return "A1b2c3"

        result = get_feature_type_from_class(TestFeature)

        # Then
        assert result == feature_enum
        assert isinstance(result, FeatureTypeEnum)


class TestFeatureTypeDecoratorEdgeCases:
    """Test edge cases and error conditions for the feature type decorator system."""

    def test_decorator_with_none_enum_value(self) -> None:
        """Test that decorator handles invalid enum values gracefully."""
        # Given/When/Then
        # This should fail at the decorator level if None is passed
        with pytest.raises(TypeError):
            @feature_type(None)  # type: ignore
            class InvalidFeature:
                pass

    def test_decorator_with_invalid_enum_type(self) -> None:
        """Test that decorator rejects non-FeatureTypeEnum values."""
        # Given/When/Then
        with pytest.raises(TypeError, match="must be a FeatureTypeEnum"):
            @feature_type("not_an_enum")  # type: ignore
            class InvalidFeature:
                pass

    def test_get_feature_type_from_class_with_none_parameter(self) -> None:
        """Test that get_feature_type_from_class handles None input gracefully."""
        # Given/When/Then
        with pytest.raises(AttributeError):
            get_feature_type_from_class(None)  # type: ignore

    def test_decorator_returns_same_class_instance(self) -> None:
        """Test that decorator returns the same class instance (not a copy)."""
        # Given
        @feature_type(FeatureTypeEnum.RSI)
        class OriginalFeature(BaseFeature):
            def compute(self) -> DataFrame:
                return DataFrame()

            def get_sub_features_names(self) -> list[str]:
                return ["original"]

            def get_feature_name(self) -> str:
                return "original"

            def get_config_to_string(self) -> str:
                return "A1b2c3"

        # When
        # Test that the decorator returns the same class, not a wrapper
        # This is important for inheritance and isinstance checks
        assert OriginalFeature.__name__ == "OriginalFeature"
        assert hasattr(OriginalFeature, '_feature_type')

    def test_decorator_works_with_multiple_classes(self) -> None:
        """Test that decorator can be applied to multiple classes independently."""
        # Given/When
        @feature_type(FeatureTypeEnum.RSI)
        class FirstFeature(BaseFeature):
            def compute(self) -> DataFrame:
                return DataFrame()

            def get_sub_features_names(self) -> list[str]:
                return ["first"]

            def get_feature_name(self) -> str:
                return "first"

            def get_config_to_string(self) -> str:
                return "A1b2c3"

        @feature_type(FeatureTypeEnum.RSI)
        class SecondFeature(BaseFeature):
            def compute(self) -> DataFrame:
                return DataFrame()

            def get_sub_features_names(self) -> list[str]:
                return ["second"]

            def get_feature_name(self) -> str:
                return "second"

            def get_config_to_string(self) -> str:
                return "A1b2c3"

        # Then
        assert FirstFeature._feature_type == FeatureTypeEnum.RSI
        assert SecondFeature._feature_type == FeatureTypeEnum.RSI
        assert FirstFeature is not SecondFeature
        assert get_feature_type_from_class(FirstFeature) == FeatureTypeEnum.RSI
        assert get_feature_type_from_class(SecondFeature) == FeatureTypeEnum.RSI
