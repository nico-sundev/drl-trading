"""
Utility for converting between FeatureTypeEnum and string representations.

This module provides type-safe conversion between strategy-specific enums
and common library string-based interfaces, solving the circular dependency issue.
"""

from drl_trading_strategy_example.enum.feature_type_enum import FeatureTypeEnum


class FeatureTypeConverter:
    """Converts between FeatureTypeEnum and string representations."""

    @staticmethod
    def enum_to_string(feature_type: FeatureTypeEnum) -> str:
        """Convert a FeatureTypeEnum to its string value.

        Args:
            feature_type: The enum to convert

        Returns:
            The string value of the enum
        """
        return feature_type.value

    @staticmethod
    def string_to_enum(feature_name: str) -> FeatureTypeEnum:
        """Convert a string to FeatureTypeEnum.

        Args:
            feature_name: The string name to convert

        Returns:
            The corresponding FeatureTypeEnum if found

        Raises:
            ValueError: If the string does not correspond to any FeatureTypeEnum
        """
        # Normalize the string to lowercase for comparison
        normalized_name = feature_name.lower().strip()

        for feature_type in FeatureTypeEnum:
            if feature_type.value == normalized_name:
                return feature_type

        raise ValueError(f"'{feature_name}' is not a valid FeatureTypeEnum value. Available types: {[e.value for e in FeatureTypeEnum]}")

    @staticmethod
    def get_all_feature_names() -> list[str]:
        """Get all available feature names as strings.

        Returns:
            List of all feature type names as strings
        """
        return [feature_type.value for feature_type in FeatureTypeEnum]

    @staticmethod
    def validate_feature_name(feature_name: str) -> bool:
        """Check if a feature name is valid.

        Args:
            feature_name: The name to validate

        Returns:
            True if the name corresponds to a valid feature type, False otherwise
        """
        try:
            FeatureTypeConverter.string_to_enum(feature_name)
            return True
        except ValueError:
            return False
