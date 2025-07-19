import logging
from typing import Dict, Optional, Type

from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_common.base.thread_safe_discoverable_registry import (
    ThreadSafeDiscoverableRegistry,
)
from drl_trading_strategy_example.decorator.feature_type_decorator import (
    get_feature_type_from_class,
)
from drl_trading_strategy_example.enum.feature_type_enum import FeatureTypeEnum
from drl_trading_strategy_example.feature.registry.feature_config_registry_interface import (
    IFeatureConfigRegistry,
)
from drl_trading_strategy_example.utils.feature_type_converter import FeatureTypeConverter

logger = logging.getLogger(__name__)


class FeatureConfigRegistry(ThreadSafeDiscoverableRegistry[FeatureTypeEnum, BaseParameterSetConfig], IFeatureConfigRegistry):
    """
    Concrete implementation of FeatureConfigRegistryInterface.

    This registry discovers, stores, and manages feature configuration class types.
    """

    def get_config_class(self, feature_type_string: str) -> Optional[Type[BaseParameterSetConfig]]:
        return self.get_class(FeatureTypeConverter.string_to_enum(feature_type_string))

    def register_config_class(
        self, feature_type_string: str, config_class: Type[BaseParameterSetConfig]
    ) -> None:
        self.register_class(FeatureTypeConverter.string_to_enum(feature_type_string), config_class)

    def discover_config_classes(
        self, package_name: str
    ) -> Dict[FeatureTypeEnum, Type[BaseParameterSetConfig]]:
        """
        Discover and register configuration classes from a specified package.
        This implementation delegates to the base class discovery method.
        """
        return self.discover_classes(package_name)

    def _validate_class(self, class_type: Type[BaseParameterSetConfig]) -> None:
        """Validate that the class extends BaseParameterSetConfig."""
        if not issubclass(class_type, BaseParameterSetConfig):
            raise TypeError(
                f"Config class {class_type.__name__} must extend BaseParameterSetConfig"
            )

    def _should_discover_class(self, class_obj) -> bool:
        """Check if a class is a valid config class for discovery."""
        return (
            issubclass(class_obj, BaseParameterSetConfig)
            and class_obj is not BaseParameterSetConfig
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
            return FeatureTypeConverter.string_to_enum(class_obj.__name__.replace("Config", "").lower())
