"""
Interface for technical indicator service operations.

This module defines the contract for technical indicator services that manage
indicator instances and provide access to their values.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict


class TechnicalIndicatorServiceInterface(ABC):
    """
    Interface defining the contract for technical indicator service operations.

    Implementations of this interface are responsible for:
    1. Managing indicator instances with unique names
    2. Updating indicator values with new data
    3. Providing access to the latest indicator values
    """

    @abstractmethod
    def register_instance(self, name: str, indicator_type: str, **params) -> None:
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
    def update(self, name: str, value: Any) -> None:
        """
        Update an indicator instance with a new value.

        Args:
            name: Name of the indicator instance to update
            value: The new value to add to the indicator

        Raises:
            KeyError: If no indicator with the given name exists
        """
        pass

    @abstractmethod
    def latest(self, name: str) -> Any:
        """
        Get the latest value from an indicator instance.

        Args:
            name: Name of the indicator instance

        Returns:
            The latest calculated value from the indicator

        Raises:
            KeyError: If no indicator with the given name exists
        """
        pass

    @property
    @abstractmethod
    def instances(self) -> Dict[str, Any]:
        """
        Get a dictionary of all registered indicator instances.

        Returns:
            Dict mapping indicator names to their instances
        """
        pass
