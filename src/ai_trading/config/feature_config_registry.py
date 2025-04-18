import importlib
import inspect
import pkgutil
from typing import Any, Dict, Type

from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig


class FeatureConfigRegistry:
    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "FeatureConfigRegistry":
        if cls._instance is None:
            cls._instance = super(FeatureConfigRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "feature_config_map"):
            self.feature_config_map = self.discover_config_classes()

    def discover_config_classes(self) -> Dict[str, Type[BaseParameterSetConfig]]:
        config_map = {}
        package_name = "ai_trading.config"
        config_package = importlib.import_module(package_name)

        for _, module_name, is_pkg in pkgutil.iter_modules(config_package.__path__):
            if is_pkg:
                continue

            full_module_name = f"{package_name}.{module_name}"
            module = importlib.import_module(full_module_name)

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, BaseParameterSetConfig)
                    and obj is not BaseParameterSetConfig
                ):
                    feature_name = name.replace("Config", "").lower()
                    config_map[feature_name] = obj

        return config_map
