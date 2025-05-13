import importlib
import inspect
import logging
import pkgutil
from typing import Dict, Type

from drl_trading_framework.preprocess.feature.collection.base_feature import BaseFeature

logger = logging.getLogger(__name__)


class FeatureClassRegistry:
    def __init__(self) -> None:
        self._feature_class_map: Dict[str, Type[BaseFeature]] = {}

    def discover_feature_classes(self) -> Dict[str, Type[BaseFeature]]:
        feature_map = {}
        package_name = "drl_trading_framework.preprocess.feature.collection"
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
                    feature_map[feature_name] = obj
                    discovered_count += 1
                    module_features_found += 1

            if module_features_found > 0:
                logger.debug(
                    f"Found {module_features_found} feature class(es) in module {full_module_name}"
                )

        logger.info(
            f"Feature class discovery complete. Found {discovered_count} feature classes"
        )
        return feature_map

    def reset(self) -> None:
        """
        Clear all registered feature classes and reset the registry state.

        This is particularly useful for testing to ensure clean isolation between tests
        and to avoid test dependencies based on registry state.
        """
        logger.debug("Resetting feature class registry")
        self._feature_class_map.clear()

    @property
    def feature_class_map(self) -> Dict[str, Type[BaseFeature]]:
        if not self._feature_class_map:
            self._feature_class_map = self.discover_feature_classes()
        return self._feature_class_map
