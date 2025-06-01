"""
Interface for indicator backend registry operations.

This module defines the contract for indicator backend registries that provide
access to different indicator implementations.
"""
from abc import ABC, abstractmethod
from typing import Any


class IndicatorBackendRegistryInterface(ABC):
    """
    Interface defining the contract for indicator backend registry operations.

    Implementations of this interface are responsible for:
    1. Providing access to different indicator implementations
    2. Supporting parameterized indicator creation
    3. Managing the mapping between indicator keys and their constructors
    """

    @abstractmethod
    def get_indicator(self, key: str, **kwargs) -> Any:
        """
        Get an indicator instance for the given key and parameters.

        Args:
            key: The identifier for the type of indicator to create
            **kwargs: Parameters to pass to the indicator constructor

        Returns:
            An instance of the requested indicator

        Raises:
            ValueError: If no indicator is registered for the given key
        """
        pass
