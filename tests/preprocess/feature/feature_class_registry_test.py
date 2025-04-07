import types
from unittest import mock

import pytest
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature
from ai_trading.preprocess.feature.feature_class_registry import FeatureClassRegistry

# Mocked Feature classes for testing
class MacdFeature(BaseFeature):
    pass

class RsiFeature(BaseFeature):
    pass

@pytest.fixture
def registry():
    return FeatureClassRegistry()

# Test the discover_feature_classes function
def test_discover_feature_classes(registry):
    with mock.patch("pkgutil.iter_modules") as mock_iter_modules, \
         mock.patch("importlib.import_module") as mock_import_module, \
         mock.patch("inspect.getmembers") as mock_getmembers:

        # Mock the base package (with __path__) and a feature module
        mock_base_package = mock.Mock()
        mock_base_package.__path__ = ["path_to_features"]
        
        # Create mock module dynamically with real class objects
        mock_feature_module = types.ModuleType("mock_feature_module")
        setattr(mock_feature_module, "MacdFeature", MacdFeature)
        setattr(mock_feature_module, "RsiFeature", RsiFeature)

        def import_module_side_effect(name):
            if name == "ai_trading.preprocess.feature.collection":
                return mock_base_package
            else:
                return mock_feature_module

        mock_import_module.side_effect = import_module_side_effect

        # You can now skip mocking `inspect.getmembers` entirely, it will work


        # Mock iter_modules to simulate one module
        mock_iter_modules.return_value = [
            ("", "macd_feature", False)
        ]

        # Now when registry accesses .feature_class_map, it will discover those features
        feature_map = registry.feature_class_map

        assert "macd" in feature_map
        assert "rsi" in feature_map
        assert feature_map["macd"] is MacdFeature
        assert feature_map["rsi"] is RsiFeature
        assert len(feature_map) == 2
        assert "base" not in feature_map

