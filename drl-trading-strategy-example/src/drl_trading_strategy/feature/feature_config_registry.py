import importlib
import inspect
import logging
import pkgutil
from typing import Dict, Optional, Type

from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_common.interfaces.feature.feature_config_registry_interface import (
    FeatureConfigRegistryInterface,
)

logger = logging.getLogger(__name__)


class FeatureConfigRegistry(FeatureConfigRegistryInterface):
    """
    Concrete implementation of FeatureConfigRegistryInterface.

    This registry discovers, stores, and manages feature configuration class types.
    It reuses the discovery logic from the original factory implementation.
    """

    def __init__(self) -> None:
        self._feature_config_map: Dict[str, Type[BaseParameterSetConfig]] = {}

    def get_config_class(self, feature_name: str) -> Optional[Type[BaseParameterSetConfig]]:
        return self._feature_config_map.get(feature_name.lower())

    def register_config_class(
        self, feature_name: str, config_class: Type[BaseParameterSetConfig]
    ) -> None:
        if not issubclass(config_class, BaseParameterSetConfig):
            raise TypeError(
                f"Config class {config_class.__name__} must extend BaseParameterSetConfig"
            )

        key = feature_name.lower()
        if key in self._feature_config_map:
            logger.warning(
                f"Overriding existing config class for feature '{key}': "
                f"{self._feature_config_map[key].__name__} -> {config_class.__name__}"
            )

        self._feature_config_map[key] = config_class
        logger.debug(f"Registered config class '{key}': {config_class}")

    def discover_config_classes(
        self, package_name: str
    ) -> Dict[str, Type[BaseParameterSetConfig]]:
        """
        Discover and register configuration classes from a specified package.
        This implementation is moved from the original FeatureConfigFactory.
        """
        logger.info(f"Starting config class discovery from package: {package_name}")
        discovered_count = 0
        processed_modules = 0

        try:
            package = importlib.import_module(package_name)
        except ImportError as e:
            logger.error(f"Could not import package {package_name}: {e}")
            return self._feature_config_map

        for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
            if is_pkg:
                continue

            full_module_name = f"{package_name}.{module_name}"
            processed_modules += 1

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
                    # Log each discovery with metadata
                    logger.debug(
                        f"Discovered config class: '{feature_name}' from class {name} "
                        f"in module {full_module_name}"
                    )
                    self.register_config_class(feature_name, obj)
                    discovered_count += 1
                    module_configs_found += 1

            if module_configs_found > 0:
                logger.debug(
                    f"Found {module_configs_found} config class(es) in module {full_module_name}"
                )

        logger.info(
            f"Config class discovery complete. Found {discovered_count} config classes across {processed_modules} modules"
        )
        return self._feature_config_map

    def reset(self) -> None:
        logger.debug("Resetting feature config registry")
        self._feature_config_map.clear()
