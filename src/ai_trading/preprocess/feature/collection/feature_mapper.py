import importlib
import inspect
import pkgutil
from typing import Dict, Type
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature

def discover_feature_classes() -> Dict[str, Type[BaseFeature]]:
    feature_map = {}
    package_name = "ai_trading.preprocess.feature.collection"
    package = importlib.import_module(package_name)

    for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
        if is_pkg:
            continue
        full_module_name = f"{package_name}.{module_name}"
        module = importlib.import_module(full_module_name)

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseFeature) and obj is not BaseFeature:
                feature_name = name.replace("Feature", "").lower()
                feature_map[feature_name] = obj

    return feature_map


FEATURE_MAP: Dict[str, Type[BaseFeature]] = discover_feature_classes()

