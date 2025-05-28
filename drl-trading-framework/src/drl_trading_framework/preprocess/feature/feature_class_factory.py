import importlib
import inspect
import logging
import pkgutil
from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

from drl_trading_framework.preprocess.feature.collection.base_feature import BaseFeature

logger = logging.getLogger(__name__)


class FeatureClassFactoryInterface(ABC):
    """
    Interface for feature class factory implementations.

    Responsible for discovering and managing feature classes.
    """

    @abstractmethod
    def get_feature_class(self, feature_name: str) -> Optional[Type[BaseFeature]]:
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
    def register_feature_class(
        self, feature_name: str, config_class: Type[BaseFeature]
    ) -> None:
        """
        Register a configuration class for a given feature name.

        Args:
            feature_name: The name of the feature (case will be normalized to lowercase)
            config_class: The configuration class to register
        """
        pass

    @abstractmethod
    def discover_feature_classes(
        self, package_name: str
    ) -> Dict[str, Type[BaseFeature]]:
        """
        Discover and register feature classes from a specified package.

        Args:
            package_name: The name of the package to discover feature classes from

        Returns:
            A dictionary mapping feature names to their corresponding class types
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Clear all registered feature classes and reset the factory state.
        """
        pass


class FeatureClassFactory(FeatureClassFactoryInterface):
    """
    Standard implementation of FeatureClassFactoryInterface.

    Discovers, registers, and manages feature classes.
    """

    def __init__(self) -> None:
        self._feature_class_map: Dict[str, Type[BaseFeature]] = {}

    def get_feature_class(self, feature_name: str) -> Optional[Type[BaseFeature]]:
        return self._feature_class_map.get(feature_name.lower())

    def register_feature_class(
        self, feature_name: str, config_class: Type[BaseFeature]
    ) -> None:
        key = feature_name.lower()
        self._feature_class_map[key] = config_class
        logger.debug(f"Registered feature class '{key}': {config_class}")

    def discover_feature_classes(
        self, package_name: str
    ) -> Dict[str, Type[BaseFeature]]:

        logger.info(f"Starting feature class discovery from package: {package_name}")
        discovered_count = 0
        package = importlib.import_module(package_name)

        for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
            if is_pkg:
                continue
            full_module_name = f"{package_name}.{module_name}"
            module = importlib.import_module(full_module_name)
            module_features_found = 0

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseFeature) and obj is not BaseFeature:
                    feature_name = name.replace("Feature", "").lower()
                    # Log each discovery with metadata
                    logger.debug(
                        f"Discovered feature class: '{feature_name}' from class {name} "
                        f"in module {full_module_name}"
                    )
                    self._feature_class_map[feature_name] = obj
                    discovered_count += 1
                    module_features_found += 1

            if module_features_found > 0:
                logger.debug(
                    f"Found {module_features_found} feature class(es) in module {full_module_name}"
                )

        logger.info(
            f"Feature class discovery complete. Found {discovered_count} feature classes"
        )
        return self._feature_class_map

    def reset(self) -> None:
        logger.debug("Resetting feature class factory")
        self._feature_class_map.clear()
