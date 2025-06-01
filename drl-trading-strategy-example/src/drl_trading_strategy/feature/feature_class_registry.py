import importlib
import inspect
import logging
import pkgutil
from typing import Dict, Optional, Type

from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.interfaces.feature.feature_class_registry_interface import (
    FeatureClassRegistryInterface,
)

logger = logging.getLogger(__name__)


class FeatureClassRegistry(FeatureClassRegistryInterface):
    """
    Concrete implementation of FeatureClassRegistryInterface.

    This registry discovers, stores, and manages feature class types.
    It reuses the discovery logic from the original factory implementation.
    """

    def __init__(self) -> None:
        self._feature_class_map: Dict[str, Type[BaseFeature]] = {}

    def get_feature_class(self, feature_name: str) -> Optional[Type[BaseFeature]]:
        return self._feature_class_map.get(feature_name.lower())

    def register_feature_class(
        self, feature_name: str, feature_class: Type[BaseFeature]
    ) -> None:
        key = feature_name.lower()
        self._feature_class_map[key] = feature_class
        logger.debug(f"Registered feature class '{key}': {feature_class}")

    def discover_feature_classes(
        self, package_name: str
    ) -> Dict[str, Type[BaseFeature]]:
        """
        Discover and register feature classes from a specified package.
        This implementation is moved from the original FeatureClassFactory.
        """
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
                # Check if the class implements BaseFeature (duck typing)
                if (
                    hasattr(obj, 'compute') and
                    hasattr(obj, 'get_sub_features_names') and
                    hasattr(obj, 'get_feature_name') and
                    obj.__name__ != 'BaseFeature'
                ):
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
        logger.debug("Resetting feature class registry")
        self._feature_class_map.clear()
