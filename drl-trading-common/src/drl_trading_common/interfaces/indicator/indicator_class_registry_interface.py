from abc import ABC, abstractmethod
from typing import Optional, Type

from drl_trading_common.base import BaseIndicator
from drl_trading_strategy.enum.indicator_type_enum import IndicatorTypeEnum


class IndicatorClassRegistryInterface(ABC):
    """
    Interface for indicator class registry implementations.

    The registry is responsible for discovering, storing, and retrieving indicator class types.
    This separates the concern of class management from instance creation.
    """

    @abstractmethod
    def get_indicator_class(self, indicator_name: IndicatorTypeEnum) -> Optional[Type[BaseIndicator]]:
        """
        Get the indicator class for a given indicator name.

        Args:
            indicator_name: The name of the indicator to get the class for (case-insensitive)

        Returns:
            The indicator class if found, None otherwise
        """
        pass

    @abstractmethod
    def register_indicator_class(
        self, indicator_name: IndicatorTypeEnum, indicator_class: Type[BaseIndicator]
    ) -> None:
        """
        Register an indicator class for a given indicator name.

        Args:
            indicator_name: The name of the indicator (case will be normalized to lowercase)
            indicator_class: The indicator class to register
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Clear all registered indicator classes and reset the registry state.
        """
        pass
