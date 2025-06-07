"""
Interface for technical indicator service operations.

This module defines the contract for technical indicator services that manage
indicator instances and provide access to their values.
"""
from abc import ABC, abstractmethod
from typing import Optional

from drl_trading_strategy.enum.indicator_type_enum import IndicatorTypeEnum
from pandas import DataFrame

class TechnicalIndicatorFacadeInterface(ABC):

    @abstractmethod
    def register_instance(self, name: str, indicator_type: IndicatorTypeEnum, **params) -> None:
        """
        Register a new indicator instance with the given name and parameters.

        Args:
            name: Unique identifier for the indicator instance
            indicator_type: Type of indicator to create (e.g., "rsi", "ema", "macd")
            **params: Parameters to pass to the indicator constructor

        Raises:
            ValueError: If an indicator with the given name already exists
        """
        pass

    @abstractmethod
    def add(self, name: str, value: DataFrame) -> None:
        """
        Incrementally compute the indicator with a new value.

        :param value: New value to update the indicator with.
        """
        pass

    @abstractmethod
    def get_all(self, name: str) -> Optional[DataFrame]:
        """
        Compute the indicator for a batch of data.

        :param data: Data to compute the indicator on.
        :return: Computed indicator values.
        """
        pass

    @abstractmethod
    def get_latest(self, name: str) -> Optional[DataFrame]:
        """
        Get the latest computed value of the indicator.

        :return: Latest indicator value.
        """
        pass

class TechnicalIndicatorFactoryInterface(ABC):
    """
    Interface defining the contract for technical indicator service factory operations.

    Implementations of this interface are responsible for:
    1. Creating instances of technical indicator services
    2. Managing the lifecycle of indicator services
    """

    @abstractmethod
    def create(self, **kwargs) -> TechnicalIndicatorFacadeInterface:
        """
        Create a new instance of a technical indicator service.

        Args:
            **kwargs: Parameters to configure the service instance

        Returns:
            An instance of TechnicalIndicatorServiceInterface
        """
        pass
