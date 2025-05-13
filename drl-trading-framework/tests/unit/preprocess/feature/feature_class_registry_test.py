import types
from typing import Dict, Type
from unittest import mock

import pytest

from drl_trading_framework.preprocess.feature.collection.base_feature import BaseFeature
from drl_trading_framework.preprocess.feature.feature_class_registry import (
    FeatureClassRegistry,
)


# Mocked Feature classes for testing
class MacdFeature(BaseFeature):
    pass


class RsiFeature(BaseFeature):
    pass


@pytest.fixture(autouse=True)
def registry() -> FeatureClassRegistry:
    return FeatureClassRegistry()


def test_discover_feature_classes(registry: FeatureClassRegistry) -> None:
    """Test that feature classes are correctly discovered and mapped."""
    with mock.patch("pkgutil.iter_modules") as mock_iter_modules, mock.patch(
        "importlib.import_module"
    ) as mock_import_module:
        # Given
        # Mock the base package (with __path__) and a feature module
        mock_base_package = mock.Mock()
        mock_base_package.__path__ = ["path_to_features"]

        # Create mock module dynamically with real class objects
        mock_feature_module = types.ModuleType("mock_feature_module")
        mock_feature_module.MacdFeature = MacdFeature
        mock_feature_module.RsiFeature = RsiFeature

        def import_module_side_effect(name: str) -> mock.Mock:
            if name == "drl_trading_framework.preprocess.feature.collection":
                return mock_base_package
            else:
                return mock_feature_module

        mock_import_module.side_effect = import_module_side_effect

        # Mock iter_modules to simulate one module
        mock_iter_modules.return_value = [("", "macd_feature", False)]

        # When
        feature_map: Dict[str, Type[BaseFeature]] = registry.feature_class_map

        # Then
        assert "macd" in feature_map
        assert "rsi" in feature_map
        assert issubclass(feature_map["macd"], BaseFeature)
        assert issubclass(feature_map["rsi"], BaseFeature)
