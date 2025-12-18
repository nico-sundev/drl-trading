"""
Feature type decorator for automatic registration of feature types.

This decorator provides a clean way to associate FeatureTypeEnum values
with feature and config classes at the class level, eliminating the need
for repetitive static get_feature_type() methods.
"""

from typing import Type, TypeVar

from drl_trading_core.core.port.base_feature import BaseFeature
from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig

from drl_trading_strategy_example.enum.feature_type_enum import FeatureTypeEnum

# Type variables for generic typing
T = TypeVar('T')


def feature_type(feature_type_enum: FeatureTypeEnum):
    """
    Decorator to associate a FeatureTypeEnum with a class.

    This decorator stores the feature type as a class attribute for
    automatic discovery by registry mechanisms.

    Args:
        feature_type_enum: The FeatureTypeEnum to associate with this class

    Returns:
        The decorated class with feature type information attached

    Raises:
        TypeError: If feature_type_enum is None or not a FeatureTypeEnum

    Example:
        @feature_type(FeatureTypeEnum.RSI)
        class RsiFeature(BaseFeature):
            # Feature type automatically available via _feature_type attribute
            pass

        @feature_type(FeatureTypeEnum.RSI)
        class RsiConfig(BaseParameterSetConfig):
            # Feature type automatically available via _feature_type attribute
            pass
    """
    if feature_type_enum is None:
        raise TypeError("feature_type_enum cannot be None")

    if not isinstance(feature_type_enum, FeatureTypeEnum):
        raise TypeError(f"feature_type_enum must be a FeatureTypeEnum, got {type(feature_type_enum)}")

    def decorator(cls: Type[T]) -> Type[T]:
        # Store the feature type as a class attribute
        cls._feature_type = feature_type_enum  # type: ignore[attr-defined]
        return cls

    return decorator


def get_feature_type_from_class(cls: Type[BaseFeature] | Type[BaseParameterSetConfig]) -> FeatureTypeEnum:
    """
    Utility function to extract feature type from a decorated class.

    This function only works with decorated classes using the @feature_type decorator.

    Args:
        cls: The class to extract the feature type from

    Returns:
        The FeatureTypeEnum associated with the class

    Raises:
        AttributeError: If the class has no feature type information
    """
    if hasattr(cls, '_feature_type'):
        return cls._feature_type

    raise AttributeError(f"Class {cls.__name__} has no feature type information. "
                        f"Use @feature_type decorator to specify the feature type.")
