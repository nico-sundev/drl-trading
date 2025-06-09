import logging
from typing import Dict, Optional, Type

from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.base.discoverable_registry import DiscoverableRegistry
from drl_trading_common.interfaces.feature.feature_class_registry_interface import (
    FeatureClassRegistryInterface,
)
from drl_trading_strategy.decorator.feature_type_decorator import (
    get_feature_type_from_class,
)
from drl_trading_strategy.enum.feature_type_enum import FeatureTypeEnum
from drl_trading_strategy.utils.feature_type_converter import FeatureTypeConverter

logger = logging.getLogger(__name__)


class FeatureClassRegistry(DiscoverableRegistry[FeatureTypeEnum, BaseFeature], FeatureClassRegistryInterface):

    """
    Concrete implementation of FeatureClassRegistryInterface.

    This registry discovers, stores, and manages feature class types using
    the DiscoverableRegistry base class for common discovery logic.
    """

    def get_feature_class(self, feature_type_string: str) -> Optional[Type[BaseFeature]]:
        return self.get_class(FeatureTypeConverter.string_to_enum(feature_type_string))

    def register_feature_class(
        self, feature_type_string: str, feature_class: Type[BaseFeature]
    ) -> None:
        self.register_class(FeatureTypeConverter.string_to_enum(feature_type_string), feature_class)

    def discover_feature_classes(
        self, package_name: str
    ) -> Dict[FeatureTypeEnum, Type[BaseFeature]]:
        """
        Discover and register feature classes from a specified package.
        This implementation delegates to the base class discovery method.
        """
        return self.discover_classes(package_name)



    def _validate_class(self, class_type: Type[BaseFeature]) -> None:
        """Validate that the class implements the BaseFeature interface."""
        # For feature classes, we use duck typing to check required methods
        if not issubclass(class_type, BaseFeature):
            raise TypeError(
                f"Feature class {class_type.__name__} must extend BaseFeature"
            )

    def _should_discover_class(self, class_obj) -> bool:
        """Check if a class is a valid feature class for discovery."""
        return (
            issubclass(class_obj, BaseFeature)
            and class_obj is not BaseFeature
        )

    def _extract_key_from_class(self, class_obj) -> FeatureTypeEnum:
        """
        Extract feature type using the decorator utility function which handles
        the @feature_type decorator.
        """
        try:
            return get_feature_type_from_class(class_obj)
        except AttributeError as e:
            logger.warning(f"Failed to get feature type from {class_obj.__name__}: {e}")
            # Fallback to the original method if no feature type information is available
            return FeatureTypeConverter.string_to_enum(class_obj.__name__.replace("Feature", "").lower())
