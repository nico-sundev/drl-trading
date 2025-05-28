# filepath: c:\Users\nico-\Documents\git\ai_trading\drl-trading-framework\tests\unit\preprocess\feature\feature_class_factory_test.py
import types
from unittest import mock

import pytest

from drl_trading_framework.preprocess.feature.collection.base_feature import BaseFeature
from drl_trading_framework.preprocess.feature.feature_class_factory import (
    FeatureClassFactory,
)


# Mocked Feature classes for testing
class MacdFeature(BaseFeature):
    pass


class RsiFeature(BaseFeature):
    pass


@pytest.fixture(autouse=True)
def factory() -> FeatureClassFactory:
    return FeatureClassFactory()


def test_register_feature_class(factory: FeatureClassFactory) -> None:
    """Test that feature classes can be manually registered."""
    # Given
    # Clean state
    factory.reset()

    # When
    factory.register_feature_class("macd", MacdFeature)
    factory.register_feature_class("RSI", RsiFeature)  # Testing case normalization

    # Then
    assert factory.get_feature_class("macd") == MacdFeature
    assert factory.get_feature_class("rsi") == RsiFeature
    assert factory.get_feature_class("MACD") == MacdFeature  # Case insensitivity
    assert factory.get_feature_class("nonexistent") is None


def test_discover_feature_classes(factory: FeatureClassFactory) -> None:
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
        package_name = "tests.unit.preprocess.feature"
        feature_map = factory.discover_feature_classes(package_name)

        # Then
        assert "macd" in feature_map
        assert feature_map == factory._feature_class_map
        assert "macd" in factory._feature_class_map
        assert factory.get_feature_class("macd") == MacdFeature
        assert issubclass(factory.get_feature_class("macd"), BaseFeature)
