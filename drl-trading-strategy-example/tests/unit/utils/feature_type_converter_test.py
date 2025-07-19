"""
Unit tests for FeatureTypeConverter utility class.

Tests the conversion between FeatureTypeEnum and string representations,
including edge cases and error handling.
"""

import pytest
from drl_trading_strategy_example.enum.feature_type_enum import FeatureTypeEnum
from drl_trading_strategy_example.utils.feature_type_converter import FeatureTypeConverter


class TestFeatureTypeConverter:
    """Test cases for FeatureTypeConverter utility class."""

    def test_enum_to_string_with_valid_enum(self) -> None:
        """Test converting a valid FeatureTypeEnum to string."""
        # Given
        feature_type = FeatureTypeEnum.RSI

        # When
        result = FeatureTypeConverter.enum_to_string(feature_type)

        # Then
        assert result == "rsi"
        assert isinstance(result, str)

    def test_string_to_enum_with_valid_string(self) -> None:
        """Test converting a valid string to FeatureTypeEnum."""
        # Given
        feature_name = "rsi"

        # When
        result = FeatureTypeConverter.string_to_enum(feature_name)

        # Then
        assert result == FeatureTypeEnum.RSI
        assert isinstance(result, FeatureTypeEnum)

    def test_string_to_enum_with_uppercase_string(self) -> None:
        """Test converting uppercase string to FeatureTypeEnum."""
        # Given
        feature_name = "RSI"

        # When
        result = FeatureTypeConverter.string_to_enum(feature_name)

        # Then
        assert result == FeatureTypeEnum.RSI
        assert isinstance(result, FeatureTypeEnum)

    def test_string_to_enum_with_mixed_case_string(self) -> None:
        """Test converting mixed case string to FeatureTypeEnum."""
        # Given
        feature_name = "RsI"

        # When
        result = FeatureTypeConverter.string_to_enum(feature_name)

        # Then
        assert result == FeatureTypeEnum.RSI
        assert isinstance(result, FeatureTypeEnum)

    def test_string_to_enum_with_whitespace(self) -> None:
        """Test converting string with whitespace to FeatureTypeEnum."""
        # Given
        feature_name = "  rsi  "

        # When
        result = FeatureTypeConverter.string_to_enum(feature_name)

        # Then
        assert result == FeatureTypeEnum.RSI
        assert isinstance(result, FeatureTypeEnum)

    def test_string_to_enum_with_invalid_string_raises_value_error(self) -> None:
        """Test that invalid string raises ValueError with helpful message."""
        # Given
        invalid_feature_name = "invalid_feature"

        # When & Then
        with pytest.raises(ValueError) as exc_info:
            FeatureTypeConverter.string_to_enum(invalid_feature_name)

        assert "invalid_feature" in str(exc_info.value)
        assert "not a valid FeatureTypeEnum value" in str(exc_info.value)
        assert "Available types: ['rsi', 'close_price']" in str(exc_info.value)

    def test_string_to_enum_with_empty_string_raises_value_error(self) -> None:
        """Test that empty string raises ValueError."""
        # Given
        empty_string = ""

        # When & Then
        with pytest.raises(ValueError) as exc_info:
            FeatureTypeConverter.string_to_enum(empty_string)

        assert "not a valid FeatureTypeEnum value" in str(exc_info.value)

    def test_string_to_enum_with_none_raises_attribute_error(self) -> None:
        """Test that None input raises AttributeError."""
        # Given
        none_input = None

        # When & Then
        with pytest.raises(AttributeError):
            FeatureTypeConverter.string_to_enum(none_input)  # type: ignore[arg-type]

    def test_get_all_feature_names_returns_correct_list(self) -> None:
        """Test that get_all_feature_names returns all enum values as strings."""
        # Given
        expected_names = ["rsi", "close_price"]

        # When
        result = FeatureTypeConverter.get_all_feature_names()

        # Then
        assert result == expected_names
        assert isinstance(result, list)
        assert all(isinstance(name, str) for name in result)

    def test_get_all_feature_names_includes_all_enum_values(self) -> None:
        """Test that get_all_feature_names includes all FeatureTypeEnum values."""
        # Given
        all_enum_values = [feature_type.value for feature_type in FeatureTypeEnum]

        # When
        result = FeatureTypeConverter.get_all_feature_names()

        # Then
        assert set(result) == set(all_enum_values)
        assert len(result) == len(FeatureTypeEnum)

    def test_validate_feature_name_with_valid_name_returns_true(self) -> None:
        """Test that validate_feature_name returns True for valid names."""
        # Given
        valid_name = "rsi"

        # When
        result = FeatureTypeConverter.validate_feature_name(valid_name)

        # Then
        assert result is True

    def test_validate_feature_name_with_uppercase_valid_name_returns_true(self) -> None:
        """Test that validate_feature_name returns True for uppercase valid names."""
        # Given
        valid_name = "RSI"

        # When
        result = FeatureTypeConverter.validate_feature_name(valid_name)

        # Then
        assert result is True

    def test_validate_feature_name_with_invalid_name_returns_false(self) -> None:
        """Test that validate_feature_name returns False for invalid names."""
        # Given
        invalid_name = "invalid_feature"

        # When
        result = FeatureTypeConverter.validate_feature_name(invalid_name)

        # Then
        assert result is False

    def test_validate_feature_name_with_empty_string_returns_false(self) -> None:
        """Test that validate_feature_name returns False for empty string."""
        # Given
        empty_string = ""

        # When
        result = FeatureTypeConverter.validate_feature_name(empty_string)

        # Then
        assert result is False

    def test_roundtrip_conversion_maintains_consistency(self) -> None:
        """Test that enum -> string -> enum conversion maintains consistency."""
        # Given
        original_enum = FeatureTypeEnum.RSI

        # When
        string_representation = FeatureTypeConverter.enum_to_string(original_enum)
        converted_back = FeatureTypeConverter.string_to_enum(string_representation)

        # Then
        assert converted_back == original_enum
        assert converted_back is original_enum

    def test_all_enum_values_can_be_converted_to_string_and_back(self) -> None:
        """Test that all FeatureTypeEnum values can be round-trip converted."""
        # Given
        all_enums = list(FeatureTypeEnum)

        # When & Then
        for enum_value in all_enums:
            # Convert to string
            string_value = FeatureTypeConverter.enum_to_string(enum_value)
            assert isinstance(string_value, str)

            # Convert back to enum
            converted_enum = FeatureTypeConverter.string_to_enum(string_value)
            assert converted_enum == enum_value
            assert isinstance(converted_enum, FeatureTypeEnum)


class TestFeatureTypeConverterStaticMethods:
    """Test that all methods are properly static and don't require instance."""

    def test_all_methods_are_static(self) -> None:
        """Test that all FeatureTypeConverter methods can be called without instance."""
        # Given
        test_enum = FeatureTypeEnum.RSI
        test_string = "rsi"

        # When & Then - All these should work without creating an instance
        string_result = FeatureTypeConverter.enum_to_string(test_enum)
        enum_result = FeatureTypeConverter.string_to_enum(test_string)
        all_names = FeatureTypeConverter.get_all_feature_names()
        validation_result = FeatureTypeConverter.validate_feature_name(test_string)

        # Verify results are correct
        assert string_result == "rsi"
        assert enum_result == FeatureTypeEnum.RSI
        assert "rsi" in all_names
        assert validation_result is True


class TestFeatureTypeConverterEdgeCases:
    """Test edge cases and potential integration issues."""

    def test_validates_feature_name_handles_internal_exception_gracefully(self) -> None:
        """Test that validate_feature_name handles internal exceptions properly."""
        # Given
        feature_name = "rsi"

        # When
        # This should not raise an exception even if string_to_enum might
        result = FeatureTypeConverter.validate_feature_name(feature_name)

        # Then
        assert result is True

    def test_error_message_includes_available_types_for_debugging(self) -> None:
        """Test that error messages include available types for debugging."""
        # Given
        invalid_name = "unknown_feature"

        # When & Then
        with pytest.raises(ValueError) as exc_info:
            FeatureTypeConverter.string_to_enum(invalid_name)

        error_message = str(exc_info.value)
        assert "Available types:" in error_message
        assert "rsi" in error_message
        assert isinstance(exc_info.value, ValueError)

    def test_string_normalization_handles_various_whitespace(self) -> None:
        """Test that string normalization handles different types of whitespace."""
        # Given
        test_cases = [
            "rsi",      # no whitespace
            " rsi",     # leading space
            "rsi ",     # trailing space
            " rsi ",    # both
            "  rsi  ",  # multiple spaces
            "\trsi\t",  # tabs
            "\nrsi\n",  # newlines
        ]

        # When & Then
        for test_case in test_cases:
            result = FeatureTypeConverter.string_to_enum(test_case)
            assert result == FeatureTypeEnum.RSI, f"Failed for input: '{test_case}'"
