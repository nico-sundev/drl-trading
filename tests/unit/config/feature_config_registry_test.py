import types
from unittest import mock
import pytest
from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig
from ai_trading.config.feature_config_registry import FeatureConfigRegistry


# Dummy subclasses for mocking
class RsiConfig(BaseParameterSetConfig):
    pass


class MacdConfig(BaseParameterSetConfig):
    pass


@pytest.fixture(autouse=True)
def reset_registry():
    FeatureConfigRegistry._instance = None


def test_discover_config_classes(reset_registry):
    with mock.patch("pkgutil.iter_modules") as mock_iter_modules, mock.patch(
        "importlib.import_module"
    ) as mock_import_module:

        # Mock iter_modules to simulate config modules
        mock_iter_modules.return_value = [
            (None, "macd_config", False),
            (None, "rsi_config", False),
        ]

        # Base package mock (with __path__)
        mock_base_package = mock.Mock()
        mock_base_package.__path__ = ["path_to_config"]

        # Create mocked feature config module
        mock_feature_module = types.ModuleType("mock_feature_module")
        setattr(mock_feature_module, "MacdConfig", MacdConfig)
        setattr(mock_feature_module, "RsiConfig", RsiConfig)

        # Dynamic import logic
        def import_module_side_effect(name):
            if name == "ai_trading.config":
                return mock_base_package
            return mock_feature_module

        mock_import_module.side_effect = import_module_side_effect

        # Instantiate after mocking
        registry = FeatureConfigRegistry()
        config_map = registry.feature_config_map

        assert "rsi" in config_map
        assert "macd" in config_map
        assert config_map["rsi"].__name__ == "RsiConfig"
        assert config_map["macd"].__name__ == "MacdConfig"
