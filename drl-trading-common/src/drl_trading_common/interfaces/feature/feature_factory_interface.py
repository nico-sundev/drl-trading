
from abc import ABC, abstractmethod
from typing import Optional


from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_common.models.dataset_identifier import DatasetIdentifier


class FeatureFactoryInterface(ABC):
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
        config: BaseParameterSetConfig,
        postfix: str = ""
    ) -> Optional[BaseFeature]:
        """
        Create a feature instance for the given feature name and parameters.

        Args:
            feature_name: The name of the feature to create
            source_data: The source data for the feature computation
            config: The configuration for the feature
            postfix: Optional postfix for the feature name
            metrics_service: Optional metrics service for the feature

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
