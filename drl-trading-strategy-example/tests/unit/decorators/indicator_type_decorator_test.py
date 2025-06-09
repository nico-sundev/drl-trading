"""
Unit tests for indicator_type_decorator module.

Tests the @indicator_type decorator and get_indicator_type_from_class utility function
to ensure proper indicator type registration and extraction.
"""

import pytest
from drl_trading_common.base.base_indicator import BaseIndicator
from drl_trading_strategy.decorator.indicator_type_decorator import (
    get_indicator_type_from_class,
    indicator_type,
)
from drl_trading_strategy.enum.indicator_type_enum import IndicatorTypeEnum
from pandas import DataFrame


class TestIndicatorTypeDecorator:
    """Test cases for the @indicator_type decorator."""

    @pytest.fixture
    def sample_indicator_enum(self) -> IndicatorTypeEnum:
        """Provide a sample IndicatorTypeEnum for testing."""
        return IndicatorTypeEnum.RSI

    def test_decorator_sets_indicator_type_attribute(self, sample_indicator_enum: IndicatorTypeEnum) -> None:
        """Test that @indicator_type decorator sets _indicator_type class attribute."""
        # Given
        # A class that will be decorated with a specific indicator type

        # When
        @indicator_type(sample_indicator_enum)
        class TestIndicator(BaseIndicator):
            def add(self, value: DataFrame) -> None:
                pass

            def get_all(self) -> DataFrame:
                return DataFrame()

            def get_latest(self) -> DataFrame:
                return DataFrame()        # Then
        assert hasattr(TestIndicator, '_indicator_type')
        assert TestIndicator._indicator_type == sample_indicator_enum  # type: ignore[attr-defined]
        assert isinstance(TestIndicator._indicator_type, IndicatorTypeEnum)  # type: ignore[attr-defined]

    def test_decorator_does_not_add_static_method(self, sample_indicator_enum: IndicatorTypeEnum) -> None:
        """Test that @indicator_type decorator only sets _indicator_type attribute (no static method)."""
        # Given
        # A class that will be decorated with a specific indicator type

        # When
        @indicator_type(sample_indicator_enum)
        class TestIndicator(BaseIndicator):
            def add(self, value: DataFrame) -> None:
                pass

            def get_all(self) -> DataFrame:
                return DataFrame()

            def get_latest(self) -> DataFrame:
                return DataFrame()        # Then
        assert hasattr(TestIndicator, '_indicator_type')
        assert TestIndicator._indicator_type == sample_indicator_enum  # type: ignore[attr-defined]
        # Verify that the decorator does NOT add a get_indicator_type method
        assert not hasattr(TestIndicator, 'get_indicator_type')

    def test_decorator_preserves_original_class_functionality(self, sample_indicator_enum: IndicatorTypeEnum) -> None:
        """Test that @indicator_type decorator preserves original class methods and attributes."""
        # Given
        # A class with specific methods and attributes that will be decorated

        # When
        @indicator_type(sample_indicator_enum)
        class TestIndicator(BaseIndicator):
            custom_attribute = "test_value"

            def __init__(self, period: int = 14) -> None:
                self.period = period

            def add(self, value: DataFrame) -> None:
                # Mock implementation
                pass

            def get_all(self) -> DataFrame:
                return DataFrame({"rsi": [30.0, 50.0, 70.0]})

            def get_latest(self) -> DataFrame:
                return DataFrame({"rsi": [70.0]})

            def custom_method(self) -> str:
                return "custom_result"

        # Then
        instance = TestIndicator(period=21)
        assert instance.custom_attribute == "test_value"
        assert instance.period == 21
        assert instance.custom_method() == "custom_result"
        result_df = instance.get_all()
        assert not result_df.empty
        assert "rsi" in result_df.columns

    def test_decorator_works_with_rsi_indicator_type(self) -> None:
        """Test that @indicator_type decorator works with RSI IndicatorTypeEnum value."""
        # Given
        # RSI indicator type to test

        # When
        @indicator_type(IndicatorTypeEnum.RSI)
        class RsiIndicator(BaseIndicator):
            def add(self, value: DataFrame) -> None:
                pass

            def get_all(self) -> DataFrame:
                return DataFrame({"rsi": [50.0]})

            def get_latest(self) -> DataFrame:
                return DataFrame({"rsi": [50.0]})        # Then
        assert RsiIndicator._indicator_type == IndicatorTypeEnum.RSI  # type: ignore[attr-defined]

    def test_decorator_works_with_inheritance(self, sample_indicator_enum: IndicatorTypeEnum) -> None:
        """Test that @indicator_type decorator works correctly with class inheritance."""
        # Given
        # A base class and a derived class, both decorated

        # When
        @indicator_type(sample_indicator_enum)
        class BaseTestIndicator(BaseIndicator):
            def add(self, value: DataFrame) -> None:
                pass

            def get_all(self) -> DataFrame:
                return DataFrame({"base": [1.0]})

            def get_latest(self) -> DataFrame:
                return DataFrame({"base": [1.0]})

        @indicator_type(IndicatorTypeEnum.RSI)
        class DerivedTestIndicator(BaseTestIndicator):
            def get_all(self) -> DataFrame:
                return DataFrame({"derived": [2.0]})

        # Then
        assert BaseTestIndicator._indicator_type == sample_indicator_enum
        assert DerivedTestIndicator._indicator_type == IndicatorTypeEnum.RSI

    def test_decorator_with_non_baseindicator_class(self, sample_indicator_enum: IndicatorTypeEnum) -> None:
        """Test that @indicator_type decorator works with non-BaseIndicator classes."""
        # Given
        # A class that doesn't inherit from BaseIndicator

        # When
        @indicator_type(sample_indicator_enum)
        class NonIndicatorClass:
            def some_method(self) -> str:
                return "test"

        # Then
        assert hasattr(NonIndicatorClass, '_indicator_type')
        assert NonIndicatorClass._indicator_type == sample_indicator_enum


class TestGetIndicatorTypeFromClass:
    """Test cases for the get_indicator_type_from_class utility function."""

    @pytest.fixture
    def sample_indicator_enum(self) -> IndicatorTypeEnum:
        """Provide a sample IndicatorTypeEnum for testing."""
        return IndicatorTypeEnum.RSI

    def test_extract_from_decorated_class(self, sample_indicator_enum: IndicatorTypeEnum) -> None:
        """Test extracting indicator type from @indicator_type decorated class."""
        # Given
        @indicator_type(sample_indicator_enum)
        class DecoratedIndicator(BaseIndicator):
            def add(self, value: DataFrame) -> None:
                pass

            def get_all(self) -> DataFrame:
                return DataFrame({"decorated": [1.0]})

            def get_latest(self) -> DataFrame:
                return DataFrame({"decorated": [1.0]})

        # When
        result = get_indicator_type_from_class(DecoratedIndicator)

        # Then
        assert result == sample_indicator_enum
        assert isinstance(result, IndicatorTypeEnum)

    def test_extract_from_undecorated_class_raises_error(self) -> None:
        """Test that extracting from undecorated class raises AttributeError."""
        # Given
        class UndecoratedIndicator(BaseIndicator):
            def add(self, value: DataFrame) -> None:
                pass

            def get_all(self) -> DataFrame:
                return DataFrame({"undecorated": [1.0]})

            def get_latest(self) -> DataFrame:
                return DataFrame({"undecorated": [1.0]})

        # When & Then
        with pytest.raises(AttributeError, match="has no indicator type information"):
            get_indicator_type_from_class(UndecoratedIndicator)

    def test_raises_attribute_error_when_no_indicator_type_info(self) -> None:
        """Test that AttributeError is raised when class has no indicator type information."""
        # Given
        class NoIndicatorTypeClass:
            def some_method(self) -> str:
                return "test"

        # When/Then
        with pytest.raises(AttributeError) as exc_info:
            get_indicator_type_from_class(NoIndicatorTypeClass)

        assert "has no indicator type information" in str(exc_info.value)
        assert "NoIndicatorTypeClass" in str(exc_info.value)

    def test_raises_attribute_error_with_proper_message(self) -> None:
        """Test that AttributeError has descriptive message with usage instructions."""
        # Given
        class TestClass:
            pass

        # When/Then
        with pytest.raises(AttributeError) as exc_info:
            get_indicator_type_from_class(TestClass)

        error_message = str(exc_info.value)
        assert "TestClass" in error_message
        assert "has no indicator type information" in error_message
        assert "@indicator_type decorator" in error_message

    def test_works_with_rsi_indicator_type_enum(self) -> None:
        """Test that utility function works with RSI IndicatorTypeEnum value."""
        # Given
        indicator_enum = IndicatorTypeEnum.RSI

        # When
        @indicator_type(indicator_enum)
        class TestIndicator(BaseIndicator):
            def add(self, value: DataFrame) -> None:
                pass

            def get_all(self) -> DataFrame:
                return DataFrame({"test": [1.0]})

            def get_latest(self) -> DataFrame:
                return DataFrame({"test": [1.0]})

        result = get_indicator_type_from_class(TestIndicator)

        # Then
        assert result == indicator_enum
        assert isinstance(result, IndicatorTypeEnum)

    def test_extract_from_class_with_manually_set_attribute(self) -> None:
        """Test extracting indicator type from class with manually set _indicator_type attribute."""
        # Given
        class ManualIndicator(BaseIndicator):
            _indicator_type = IndicatorTypeEnum.RSI

            def add(self, value: DataFrame) -> None:
                pass

            def get_all(self) -> DataFrame:
                return DataFrame({"manual": [1.0]})

            def get_latest(self) -> DataFrame:
                return DataFrame({"manual": [1.0]})

        # When
        result = get_indicator_type_from_class(ManualIndicator)

        # Then
        assert result == IndicatorTypeEnum.RSI


class TestIndicatorTypeDecoratorEdgeCases:
    """Test edge cases and error conditions for the indicator type decorator system."""

    def test_decorator_with_none_enum_value(self) -> None:
        """Test that decorator handles invalid enum values gracefully."""
        # Given/When/Then
        # This should fail at the decorator level if None is passed
        with pytest.raises(TypeError, match="indicator_type_enum cannot be None"):
            @indicator_type(None)  # type: ignore
            class InvalidIndicator:
                pass

    def test_decorator_with_invalid_enum_type(self) -> None:
        """Test that decorator rejects non-IndicatorTypeEnum values."""
        # Given/When/Then
        with pytest.raises(TypeError, match="must be a IndicatorTypeEnum"):
            @indicator_type("not_an_enum")  # type: ignore
            class InvalidIndicator:
                pass

    def test_decorator_with_invalid_integer_type(self) -> None:
        """Test that decorator rejects integer values."""
        # Given/When/Then
        with pytest.raises(TypeError, match="must be a IndicatorTypeEnum"):
            @indicator_type(42)  # type: ignore
            class InvalidIndicator:
                pass

    def test_get_indicator_type_from_class_with_none_parameter(self) -> None:
        """Test that get_indicator_type_from_class handles None input gracefully."""
        # Given/When/Then
        with pytest.raises(AttributeError):
            get_indicator_type_from_class(None)  # type: ignore

    def test_decorator_returns_same_class_instance(self) -> None:
        """Test that decorator returns the same class instance (not a copy)."""
        # Given
        @indicator_type(IndicatorTypeEnum.RSI)
        class OriginalIndicator(BaseIndicator):
            def add(self, value: DataFrame) -> None:
                pass

            def get_all(self) -> DataFrame:
                return DataFrame({"original": [1.0]})

            def get_latest(self) -> DataFrame:
                return DataFrame({"original": [1.0]})

        # When
        # Test that the decorator returns the same class, not a wrapper
        # This is important for inheritance and isinstance checks
        assert OriginalIndicator.__name__ == "OriginalIndicator"
        assert hasattr(OriginalIndicator, '_indicator_type')

    def test_decorator_works_with_multiple_classes(self) -> None:
        """Test that decorator can be applied to multiple classes independently."""
        # Given/When
        @indicator_type(IndicatorTypeEnum.RSI)
        class FirstIndicator(BaseIndicator):
            def add(self, value: DataFrame) -> None:
                pass

            def get_all(self) -> DataFrame:
                return DataFrame({"first": [1.0]})

            def get_latest(self) -> DataFrame:
                return DataFrame({"first": [1.0]})

        @indicator_type(IndicatorTypeEnum.RSI)
        class SecondIndicator(BaseIndicator):
            def add(self, value: DataFrame) -> None:
                pass

            def get_all(self) -> DataFrame:
                return DataFrame({"second": [2.0]})

            def get_latest(self) -> DataFrame:
                return DataFrame({"second": [2.0]})

        # Then
        assert FirstIndicator._indicator_type == IndicatorTypeEnum.RSI
        assert SecondIndicator._indicator_type == IndicatorTypeEnum.RSI
        assert FirstIndicator is not SecondIndicator
        assert get_indicator_type_from_class(FirstIndicator) == IndicatorTypeEnum.RSI
        assert get_indicator_type_from_class(SecondIndicator) == IndicatorTypeEnum.RSI

    def test_decorator_preserves_class_metadata(self) -> None:
        """Test that decorator preserves important class metadata."""
        # Given
        @indicator_type(IndicatorTypeEnum.RSI)
        class DocumentedIndicator(BaseIndicator):
            """This is a documented indicator class."""

            def add(self, value: DataFrame) -> None:
                """Add a value to the indicator."""
                pass

            def get_all(self) -> DataFrame:
                """Get all computed values."""
                return DataFrame({"documented": [1.0]})

            def get_latest(self) -> DataFrame:
                """Get the latest computed value."""
                return DataFrame({"documented": [1.0]})

        # Then
        assert DocumentedIndicator.__name__ == "DocumentedIndicator"
        assert DocumentedIndicator.__doc__ == "This is a documented indicator class."
        assert DocumentedIndicator.add.__doc__ == "Add a value to the indicator."
        assert hasattr(DocumentedIndicator, '_indicator_type')
        assert DocumentedIndicator._indicator_type == IndicatorTypeEnum.RSI

    def test_decorator_type_safety_with_type_var(self) -> None:
        """Test that decorator maintains type safety with TypeVar."""
        # Given
        @indicator_type(IndicatorTypeEnum.RSI)
        class TypedIndicator(BaseIndicator):
            def __init__(self, period: int) -> None:
                self.period = period

            def add(self, value: DataFrame) -> None:
                pass

            def get_all(self) -> DataFrame:
                return DataFrame({"typed": [float(self.period)]})

            def get_latest(self) -> DataFrame:
                return DataFrame({"typed": [float(self.period)]})

        # When
        instance = TypedIndicator(14)

        # Then
        assert isinstance(instance, TypedIndicator)
        assert isinstance(instance, BaseIndicator)
        assert instance.period == 14
        assert TypedIndicator._indicator_type == IndicatorTypeEnum.RSI

    def test_decorator_with_complex_class_hierarchy(self) -> None:
        """Test decorator behavior with complex inheritance hierarchies."""
        # Given
        class AbstractIndicator(BaseIndicator):
            """Abstract base for indicators."""

            def common_method(self) -> str:
                return "common"

        @indicator_type(IndicatorTypeEnum.RSI)
        class ConcreteIndicator(AbstractIndicator):
            """Concrete implementation."""

            def add(self, value: DataFrame) -> None:
                pass

            def get_all(self) -> DataFrame:
                return DataFrame({"concrete": [1.0]})

            def get_latest(self) -> DataFrame:
                return DataFrame({"concrete": [1.0]})

        # When
        instance = ConcreteIndicator()

        # Then
        assert isinstance(instance, ConcreteIndicator)
        assert isinstance(instance, AbstractIndicator)
        assert isinstance(instance, BaseIndicator)
        assert instance.common_method() == "common"
        assert ConcreteIndicator._indicator_type == IndicatorTypeEnum.RSI
        assert get_indicator_type_from_class(ConcreteIndicator) == IndicatorTypeEnum.RSI
