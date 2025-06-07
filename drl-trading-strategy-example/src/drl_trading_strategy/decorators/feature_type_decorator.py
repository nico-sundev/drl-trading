"""
Feature type decorator for automatic registration of feature types.

This decorator provides a clean way to associate FeatureTypeEnum values
with feature and config classes at the class level, eliminating the need
for repetitive static get_feature_type() methods.
"""

from typing import Type, TypeVar

from drl_trading_strategy.enum.feature_type_enum import FeatureTypeEnum

# Type variables for generic typing
T = TypeVar('T')


def feature_type(feature_type_enum: FeatureTypeEnum):
    """
    Decorator to associate a FeatureTypeEnum with a class.

    This decorator:
    1. Stores the feature type as a class attribute
    2. Adds a static get_feature_type() method to the class
    3. Maintains compatibility with existing registry discovery mechanisms

    Args:
        feature_type_enum: The FeatureTypeEnum to associate with this class

    Returns:
        The decorated class with feature type information attached

    Example:
        @feature_type(FeatureTypeEnum.RSI)
        class RsiFeature(BaseFeature):
            # No need for get_feature_type() method anymore
            pass

        @feature_type(FeatureTypeEnum.RSI)
        class RsiConfig(BaseParameterSetConfig):
            # No need for get_feature_type() method anymore
            pass
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Store the feature type as a class attribute
        cls._feature_type = feature_type_enum

        # Add the static get_feature_type method for backwards compatibility
        @staticmethod
        def get_feature_type() -> FeatureTypeEnum:
            return feature_type_enum

        cls.get_feature_type = get_feature_type

        return cls

    return decorator


def get_feature_type_from_class(cls: Type) -> FeatureTypeEnum:
    """
    Utility function to extract feature type from a decorated class.

    This function works with both decorated classes (using _feature_type attribute)
    and classes with traditional get_feature_type() static methods.

    Args:
        cls: The class to extract the feature type from

    Returns:
        The FeatureTypeEnum associated with the class

    Raises:
        AttributeError: If the class has no feature type information
    """
    # Try the decorator approach first
    if hasattr(cls, '_feature_type'):
        return cls._feature_type

    # Fall back to the traditional static method approach
    if hasattr(cls, 'get_feature_type') and callable(cls.get_feature_type):
        return cls.get_feature_type()

    raise AttributeError(f"Class {cls.__name__} has no feature type information. "
                        f"Use @feature_type decorator or implement get_feature_type() static method.")
