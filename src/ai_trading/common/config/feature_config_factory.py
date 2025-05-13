"""
Feature configuration factory for managing configuration classes.

This module provides interfaces and implementations for feature configuration discovery
and instantiation, replacing the previous singleton registry pattern with a more
explicit and testable factory approach.
"""

import importlib
import inspect
import logging
import pkgutil  # Added missing pkgutil import
from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

from ai_trading.common.config.base_parameter_set_config import BaseParameterSetConfig

logger = logging.getLogger(__name__)


class FeatureConfigFactoryInterface(ABC):
    """
    Interface for feature configuration factory implementations.

    Responsible for discovering, registering, and creating feature configuration
    instances based on feature names.
    """

    @abstractmethod
    def get_config_class(
        self, feature_name: str
    ) -> Optional[Type[BaseParameterSetConfig]]:
        """
        Get the configuration class for a given feature name.

        Args:
            feature_name: The name of the feature to get the config class for
                          (case-insensitive)

        Returns:
            The configuration class for the feature if found, None otherwise
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
    def discover_config_classes(
        self, package_name: str = "ai_trading.common.config"
    ) -> None:
        """
        Discover and register all config classes in the specified package.

        Args:
            package_name: The Python package name to search for config classes
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
    def clear(self) -> None:
        """
        Clear all registered configuration classes.

        Useful for testing and for explicitly refreshing the configuration registry.
        """
        pass


class FeatureConfigFactory(FeatureConfigFactoryInterface):
    """
    Standard implementation of FeatureConfigFactoryInterface.

    Discovers, registers, and creates feature configuration classes based on
    feature names. Provides methods for auto-discovery as well as manual registration.
    """

    def __init__(self) -> None:
        """Initialize an empty feature config factory."""
        self._feature_config_map: Dict[str, Type[BaseParameterSetConfig]] = {}

    def get_config_class(
        self, feature_name: str
    ) -> Optional[Type[BaseParameterSetConfig]]:
        """
        Get the configuration class for a given feature name.

        Args:
            feature_name: The name of the feature to get the config class for
                          (case-insensitive)

        Returns:
            The configuration class for the feature if found, None otherwise
        """
        return self._feature_config_map.get(feature_name.lower())

    def register_config_class(
        self, feature_name: str, config_class: Type[BaseParameterSetConfig]
    ) -> None:
        """
        Register a configuration class for a given feature name.

        Args:
            feature_name: The name of the feature (case will be normalized to lowercase)
            config_class: The configuration class to register
        """
        if not issubclass(config_class, BaseParameterSetConfig):
            raise TypeError(
                f"Config class {config_class.__name__} must extend BaseParameterSetConfig"
            )

        normalized_name = feature_name.lower()
        if normalized_name in self._feature_config_map:
            logger.warning(
                f"Overriding existing config class for feature '{normalized_name}': "
                f"{self._feature_config_map[normalized_name].__name__} -> {config_class.__name__}"
            )

        self._feature_config_map[normalized_name] = config_class
        logger.debug(
            f"Registered config class {config_class.__name__} for feature '{normalized_name}'"
        )

    def discover_config_classes(
        self, package_name: str = "ai_trading.common.config"
    ) -> None:
        """
        Discover and register all config classes in the specified package.

        Uses module introspection to find classes that extend BaseParameterSetConfig.
        Class names are normalized by removing "Config" suffix and converting to lowercase.

        Args:
            package_name: The Python package name to search for config classes
        """
        discovered_count = 0
        processed_modules = 0

        try:
            package = importlib.import_module(package_name)
        except ImportError as e:
            logger.error(f"Failed to import package {package_name}: {e}")
            return

        # Use pkgutil.iter_modules to correctly iterate through package modules
        for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
            if is_pkg:
                continue

            processed_modules += 1
            full_module_name = f"{package_name}.{module_name}"

            try:
                module = importlib.import_module(full_module_name)
            except ImportError as e:
                logger.warning(f"Failed to import module {full_module_name}: {e}")
                continue

            module_configs_found = 0
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, BaseParameterSetConfig)
                    and obj is not BaseParameterSetConfig
                ):
                    feature_name = name.replace("Config", "").lower()
                    self.register_config_class(feature_name, obj)
                    discovered_count += 1
                    module_configs_found += 1

            if module_configs_found > 0:
                logger.debug(
                    f"Found {module_configs_found} config(s) in module {full_module_name}"
                )

        logger.info(
            f"Discovered and registered {discovered_count} feature config classes across {processed_modules} modules"
        )

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

    def clear(self) -> None:
        """
        Clear all registered configuration classes.

        Useful for testing and for explicitly refreshing the configuration registry.
        """
        logger.debug("Clearing all registered feature config classes")
        self._feature_config_map.clear()
