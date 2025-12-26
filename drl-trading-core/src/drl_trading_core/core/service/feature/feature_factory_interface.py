
from abc import ABC, abstractmethod
from typing import Optional


from drl_trading_core.core.port.base_feature import BaseFeature
from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_common.core.model.dataset_identifier import DatasetIdentifier


class IFeatureFactory(ABC):
    """
    Interface for feature factory implementations.

    The factory is responsible for creating feature instances using the registry
    to obtain class types. This implements the actual Factory pattern.
    """

    @abstractmethod
    def create_feature(
        self,
        feature_name: str,
        dataset_id: DatasetIdentifier,
        config: Optional[BaseParameterSetConfig] = None,
        postfix: str = ""
    ) -> Optional[BaseFeature]:
        """
        Create a feature instance for the given feature name and parameters.

        Args:
            feature_name: The name of the feature to create
            dataset_id: The dataset identifier for the feature
            config: The configuration for the feature (optional)
            postfix: Optional postfix for the feature name

        Returns:
            The created feature instance if successful, None otherwise
        """
        pass

    @abstractmethod
    def create_config_instance(
        self, feature_name: str, config_data: dict
    ) -> Optional[BaseParameterSetConfig]:
        """
        Create a configuration instance for the given feature name and data.

        Args:
            feature_name: The name of the feature to create config for
            config_data: The configuration data to initialize the config with

        Returns:
            A configuration instance or None if no config class is found

        Raises:
            ValueError: If the provided config_data is invalid for the config class
        """
        pass

    @abstractmethod
    def is_feature_supported(self, feature_name: str) -> bool:
        """
        Check if a feature is fully supported (both class and config available).

        Args:
            feature_name: The name of the feature to check

        Returns:
            True if the feature can be created successfully, False otherwise
        """
        pass
