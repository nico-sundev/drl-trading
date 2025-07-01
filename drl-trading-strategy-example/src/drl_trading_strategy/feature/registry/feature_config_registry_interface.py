from abc import ABC, abstractmethod
from typing import Optional, Type

from drl_trading_common.base import BaseParameterSetConfig


class IFeatureConfigRegistry(ABC):
    """
    Interface for feature configuration registry implementations.

    The registry is responsible for discovering, storing, and retrieving feature configuration class types.
    This separates the concern of config class management from instance creation.
    """

    @abstractmethod
    def get_config_class(self, feature_name: str) -> Optional[Type[BaseParameterSetConfig]]:
        """
        Get the configuration class for a given feature name.

        Args:
            feature_name: The name of the feature to get the config class for (case-insensitive)

        Returns:
            The configuration class if found, None otherwise
        """
        pass

    @abstractmethod
    def register_config_class(
        self, feature_name: str, config_class: Type[BaseParameterSetConfig]
    ) -> None:
        """
        Register a configuration class for a given feature name.

        Args:
            feature_name: The name of the feature (case will be normalized to lowercase)
            config_class: The configuration class to register
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Clear all registered configuration classes and reset the registry state.
        """
        pass
