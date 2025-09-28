from abc import ABC, abstractmethod
from typing import Optional, Type

from drl_trading_common.base import BaseFeature


class IFeatureClassRegistry(ABC):
    """
    Interface for feature class registry implementations.

    The registry is responsible for discovering, storing, and retrieving feature class types.
    This separates the concern of class management from instance creation.
    """

    @abstractmethod
    def get_feature_class(self, feature_name: str) -> Optional[Type[BaseFeature]]:
        """
        Get the feature class for a given feature name.

        Args:
            feature_name: The name of the feature to get the class for (case-insensitive)

        Returns:
            The feature class if found, None otherwise
        """
        pass

    @abstractmethod
    def register_feature_class(
        self, feature_name: str, feature_class: Type[BaseFeature]
    ) -> None:
        """
        Register a feature class for a given feature name.

        Args:
            feature_name: The name of the feature (case will be normalized to lowercase)
            feature_class: The feature class to register
        """
        pass

    @abstractmethod
    def has_feature_class(self, feature_name: str) -> bool:
        """
        Check if a feature class is registered for the given feature name.

        Args:
            feature_name: The name of the feature to check (case-insensitive)

        Returns:
            True if the feature class is registered, False otherwise
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Clear all registered feature classes and reset the registry state.
        """
        pass
