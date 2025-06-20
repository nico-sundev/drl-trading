
from typing import Type, TypeVar, TYPE_CHECKING

from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum

if TYPE_CHECKING:
    from drl_trading_common.base.base_feature import BaseFeature

# Type variables for generic typing
T = TypeVar('T')


def feature_role(feature_role_enum: FeatureRoleEnum):
    if feature_role_enum is None:
        raise TypeError("feature_role_enum cannot be None")

    if not isinstance(feature_role_enum, FeatureRoleEnum):
        raise TypeError(f"feature_role_enum must be a FeatureRoleEnum, got {type(feature_role_enum)}")

    def decorator(cls: Type[T]) -> Type[T]:
        # Store the feature type as a class attribute
        cls._feature_role = feature_role_enum
        return cls

    return decorator


def get_feature_role_from_class(cls: Type["BaseFeature"]) -> FeatureRoleEnum:
    """
    Utility function to extract feature role from a decorated class.

    This function only works with decorated classes using the @feature_role decorator.

    Args:
        cls: The class to extract the feature role from

    Returns:
        The FeatureRoleEnum associated with the class

    Raises:
        AttributeError: If the class has no feature role information
    """
    if hasattr(cls, '_feature_role'):
        return cls._feature_role

    raise AttributeError(f"Class {cls.__name__} has no feature role information. "
                        f"Use @feature_role decorator to specify the feature role.")
