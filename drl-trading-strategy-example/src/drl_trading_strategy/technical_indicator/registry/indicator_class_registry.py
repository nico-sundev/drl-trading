import logging
from typing import Dict, Optional, Type

from drl_trading_common.base.base_indicator import BaseIndicator
from drl_trading_common.base.thread_safe_discoverable_registry import (
    ThreadSafeDiscoverableRegistry,
)
from drl_trading_strategy.decorator.indicator_type_decorator import (
    get_indicator_type_from_class,
)
from drl_trading_strategy.enum.indicator_type_enum import IndicatorTypeEnum

logger = logging.getLogger(__name__)


class IndicatorClassRegistry(ThreadSafeDiscoverableRegistry[IndicatorTypeEnum, BaseIndicator]):
    """
    This registry discovers, stores, and manages indicator class types using
    the DiscoverableRegistry base class for common discovery logic.
    """

    def get_indicator_class(self, indicator_type: IndicatorTypeEnum) -> Optional[Type[BaseIndicator]]:
        return self.get_class(indicator_type)

    def register_indicator_class(
        self, indicator_type: IndicatorTypeEnum, indicator_class: Type[BaseIndicator]
    ) -> None:
        self.register_class(indicator_type, indicator_class)

    def discover_indicator_classes(
        self, package_name: str
    ) -> Dict[IndicatorTypeEnum, Type[BaseIndicator]]:
        """
        Discover and register indicator classes from a specified package.
        This implementation delegates to the base class discovery method.
        """
        return self.discover_classes(package_name)

    def _validate_class(self, class_type: Type[BaseIndicator]) -> None:
        """Validate that the class implements the BaseIndicator interface."""
        # For indicator classes, we use duck typing to check required methods
        if not issubclass(class_type, BaseIndicator):
            raise TypeError(
                f"Indicator class {class_type.__name__} must extend BaseIndicator"
            )

    def _should_discover_class(self, class_obj) -> bool:
        """Check if a class is a valid indicator class for discovery."""
        return (
            issubclass(class_obj, BaseIndicator)
            and class_obj is not BaseIndicator
        )

    def _extract_key_from_class(self, class_obj) -> IndicatorTypeEnum:
        """
        Extract indicator type using the decorator utility function which handles
        the @indicator_type decorator.
        """
        return get_indicator_type_from_class(class_obj)
