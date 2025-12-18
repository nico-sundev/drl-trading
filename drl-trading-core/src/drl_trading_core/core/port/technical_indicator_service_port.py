"""
Interface for technical indicator service operations.

This module defines the contract for technical indicator services that manage
indicator instances and provide access to their values.

This interface uses string-based indicator types to avoid circular dependencies
between common and strategy packages. Strategy-specific implementations can
internally convert strings to enums as needed.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional

from pandas import DataFrame


class ITechnicalIndicatorServicePort(ABC):
    """
    Generic technical indicator interface with no strategy dependencies.

    Uses string-based indicator types for decoupling, allowing strategy layers
    to handle concrete type mappings internally.
    """

    @abstractmethod
    def register_instance(self, name: str, indicator_type: str, **params: Any) -> None:
        """
        Register a new indicator instance with the given name and parameters.

        Args:
            name: Unique identifier for the indicator instance
            indicator_type: String identifier for indicator type (e.g., "rsi", "ema", "macd")
            **params: Parameters to pass to the indicator constructor

        Raises:
            ValueError: If an indicator with the given name already exists
            ValueError: If indicator_type is not supported
        """
        pass

    @abstractmethod
    def add(self, name: str, value: DataFrame) -> None:
        """
        Incrementally compute the indicator with a new value.

        Args:
            name: Name of the registered indicator instance
            value: New data to update the indicator with

        Raises:
            ValueError: If indicator with given name is not registered
        """
        pass

    @abstractmethod
    def get_all(self, name: str) -> Optional[DataFrame]:
        """
        Get all computed values for the indicator.

        Args:
            name: Name of the registered indicator instance

        Returns:
            DataFrame with all computed indicator values, or None if no data

        Raises:
            ValueError: If indicator with given name is not registered
        """
        pass

    @abstractmethod
    def get_latest(self, name: str) -> Optional[DataFrame]:
        """
        Get the latest computed value of the indicator.

        Args:
            name: Name of the registered indicator instance

        Returns:
            DataFrame with the latest indicator value, or None if no data

        Raises:
            ValueError: If indicator with given name is not registered
        """
        pass
