import logging
from abc import ABC, abstractmethod
from typing import Optional, Type

from drl_trading_common import (
    BaseParameterSetConfig,
    FeatureClassRegistryInterface,
    FeatureConfigRegistryInterface,
)
from drl_trading_common.base import BaseFeature
from drl_trading_common.interfaces.indicator.technical_indicator_facade_interface import (
    TechnicalIndicatorFacadeInterface,
)
from injector import inject

logger = logging.getLogger(__name__)


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
        source_data,
        config: BaseParameterSetConfig,
        indicators_service: TechnicalIndicatorFacadeInterface,
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
    def get_registry(self) -> FeatureClassRegistryInterface:
        """
        Get the underlying feature class registry.

        Returns:
            The feature class registry used by this factory
        """
        pass

    @abstractmethod
    def get_config_registry(self) -> FeatureConfigRegistryInterface:
        """
        Get the underlying feature config registry.

        Returns:
            The feature config registry used by this factory
        """
        pass

    @abstractmethod
    def get_config_class(self, feature_name: str) -> Optional[Type[BaseParameterSetConfig]]:
        """
        Get the configuration class for a given feature name.

        Args:
            feature_name: The name of the feature to get the config class for

        Returns:
            The configuration class if found, None otherwise
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


class FeatureFactory(FeatureFactoryInterface):
    """
    Concrete implementation of FeatureFactoryInterface.

    This factory creates feature instances using a registry to obtain class types.
    It implements the actual Factory pattern by creating instances rather than just
    managing class types.
    """

    @inject
    def __init__(
        self,
        registry: FeatureClassRegistryInterface,
        config_registry: FeatureConfigRegistryInterface
    ) -> None:
        """
        Initialize the factory with feature class and config registries.

        Args:
            registry: The feature class registry to use for obtaining class types
            config_registry: The feature config registry to use for obtaining config class types
        """
        self._registry = registry
        self._config_registry = config_registry

    def create_feature(
        self,
        feature_name: str,
        source_data,
        config: BaseParameterSetConfig,
        indicators_service: TechnicalIndicatorFacadeInterface,
        postfix: str = "",
    ) -> Optional[BaseFeature]:
        """
        Create a feature instance for the given feature name and parameters.

        Args:
            feature_name: The name of the feature to create
            source_data: The source data for the feature computation
            config: The configuration for the feature
            postfix: Optional postfix for the feature name
            indicators_service: Indicators service for the feature

        Returns:
            The created feature instance if successful, None otherwise
        """
        feature_class = self._registry.get_feature_class(feature_name)
        if feature_class is None:
            logger.error(
                f"Feature class for '{feature_name}' not found in registry. Cannot create instance."
            )
            return None

        try:
            # Create the feature instance with the provided parameters
            feature_instance = feature_class(
                source=source_data,
                config=config,
                indicator_service=indicators_service,
                postfix=postfix,
            )
            logger.debug(f"Created feature instance for '{feature_name}'")
            return feature_instance
        except Exception as e:
            logger.error(
                f"Failed to create feature instance for '{feature_name}': {e}",
                exc_info=True,
            )
            return None

    def get_registry(self) -> FeatureClassRegistryInterface:
        """
        Get the underlying feature class registry.

        Returns:
            The feature class registry used by this factory
        """
        return self._registry

    def get_config_registry(self) -> FeatureConfigRegistryInterface:
        """
        Get the underlying feature config registry.

        Returns:
            The feature config registry used by this factory
        """
        return self._config_registry

    def get_config_class(self, feature_name: str) -> Optional[Type[BaseParameterSetConfig]]:
        """
        Get the configuration class for a given feature name.

        Args:
            feature_name: The name of the feature to get the config class for

        Returns:
            The configuration class if found, None otherwise
        """
        return self._config_registry.get_config_class(feature_name)

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
        config_class = self.get_config_class(feature_name)
        if not config_class:
            logger.warning(f"No config class found for feature '{feature_name}'")
            return None

        try:
            # Create a copy of config_data to avoid modifying the original
            instance_data = config_data.copy()
            # Add type field for discriminated unions if not present
            if "type" not in instance_data:
                instance_data["type"] = feature_name.lower()

            return config_class(**instance_data)
        except Exception as e:
            error_msg = (
                f"Failed to create config instance for '{feature_name}': {str(e)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from e
