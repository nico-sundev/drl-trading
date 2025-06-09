"""
Feature type decorator for automatic registration of indicator types.

This decorator provides a clean way to associate IndicatorTypeEnum values
with indicator classes at the class level.
"""

from typing import Type, TypeVar

from drl_trading_common.base.base_indicator import BaseIndicator

from drl_trading_strategy.enum.indicator_type_enum import IndicatorTypeEnum

# Type variables for generic typing
T = TypeVar('T')


def indicator_type(indicator_type_enum: IndicatorTypeEnum):
    """
    Decorator to associate a IndicatorTypeEnum with a class.

    This decorator stores the indicator type as a class attribute for
    automatic discovery by registry mechanisms.

    Args:
        indicator_type_enum: The IndicatorTypeEnum to associate with this class

    Returns:
        The decorated class with indicator type information attached

    Raises:
        TypeError: If indicator_type_enum is None or not a IndicatorTypeEnum

    Example:
        @indicator_type(IndicatorTypeEnum.RSI)
        class RsiFeature(BaseFeature):
            # Feature type automatically available via _indicator_type attribute
            pass

        @indicator_type(IndicatorTypeEnum.RSI)
        class RsiConfig(BaseParameterSetConfig):
            # Feature type automatically available via _indicator_type attribute
            pass
    """
    if indicator_type_enum is None:
        raise TypeError("indicator_type_enum cannot be None")

    if not isinstance(indicator_type_enum, IndicatorTypeEnum):
        raise TypeError(f"indicator_type_enum must be a IndicatorTypeEnum, got {type(indicator_type_enum)}")

    def decorator(cls: Type[T]) -> Type[T]:
        # Store the indicator type as a class attribute
        cls._indicator_type = indicator_type_enum  # type: ignore[attr-defined]
        return cls

    return decorator


def get_indicator_type_from_class(cls: Type[BaseIndicator]) -> IndicatorTypeEnum:
    """
    Utility function to extract indicator type from a decorated class.

    This function only works with decorated classes using the @indicator_type decorator.

    Args:
        cls: The class to extract the indicator type from

    Returns:
        The IndicatorTypeEnum associated with the class

    Raises:
        AttributeError: If the class has no indicator type information
    """
    if hasattr(cls, '_indicator_type'):
        return cls._indicator_type

    raise AttributeError(f"Class {cls.__name__} has no indicator type information. "
                        f"Use @indicator_type decorator to specify the indicator type.")
