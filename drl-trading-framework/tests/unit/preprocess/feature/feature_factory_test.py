from unittest.mock import MagicMock

import pandas as pd
import pytest
from drl_trading_common import BaseParameterSetConfig, FeatureConfigRegistryInterface
from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.interfaces.feature.feature_class_registry_interface import (
    FeatureClassRegistryInterface,
)

from drl_trading_framework.preprocess.feature.feature_factory import FeatureFactory


# Mocked Feature classes for testing
class MacdFeature(BaseFeature):
    def __init__(self, source, config, postfix="", metrics_service=None):
        super().__init__(source, config, postfix, metrics_service)

    def compute(self):
        return pd.DataFrame()

    def get_sub_features_names(self):
        return ["macd"]


class RsiFeature(BaseFeature):
    def __init__(self, source, config, postfix="", metrics_service=None):
        super().__init__(source, config, postfix, metrics_service)

    def compute(self):
        return pd.DataFrame()

    def get_sub_features_names(self):
        return ["rsi"]


class MockConfig(BaseParameterSetConfig):
    type: str = "test"
    enabled: bool = True


class RsiConfig(BaseParameterSetConfig):
    type: str = "rsi"
    length: int = 14
    enabled: bool = True


class MacdConfig(BaseParameterSetConfig):
    type: str = "macd"
    fast: int = 12
    slow: int = 26
    signal: int = 9
    enabled: bool = True


@pytest.fixture
def class_registry() -> FeatureClassRegistryInterface:
    """Create a mock feature class registry that returns RsiFeature for 'rsi' type."""
    mock = MagicMock(spec=FeatureClassRegistryInterface)    # Define mapping of feature types to feature classes
    feature_mapping = {
        "rsi": RsiFeature,
        "macd": MacdFeature,
    }

    def get_feature_class(feature_type: str):
        if feature_type in feature_mapping:
            return feature_mapping[feature_type]
        return None  # Return None for unknown feature types

    mock.get_feature_class.side_effect = get_feature_class
    mock.reset.side_effect = lambda: None  # Reset method does nothing
    return mock


@pytest.fixture
def config_registry() -> FeatureConfigRegistryInterface:
    """Create a mock feature config registry."""
    mock = MagicMock(spec=FeatureConfigRegistryInterface)
      # Define mapping of feature types to config classes
    config_mapping = {
        "rsi": RsiConfig,
        "macd": MacdConfig,
        "test": MockConfig,
    }

    def get_config_class(feature_type: str):
        return config_mapping.get(feature_type.lower())

    mock.get_config_class.side_effect = get_config_class
    mock.reset.side_effect = lambda: None
    return mock


@pytest.fixture
def factory(class_registry, config_registry) -> FeatureFactory:
    return FeatureFactory(class_registry, config_registry)


def test_create_feature(factory: FeatureFactory) -> None:
    """Test that feature instances can be created using the factory."""    # Given
    source_data = pd.DataFrame({"Close": [1, 2, 3, 4, 5]})
    config = MockConfig()

    # When & Then
    macd_feature = factory.create_feature("macd", source_data, config)
    assert macd_feature is not None
    assert isinstance(macd_feature, MacdFeature)

    rsi_feature = factory.create_feature("rsi", source_data, config)
    assert rsi_feature is not None
    assert isinstance(rsi_feature, RsiFeature)

    # Test with postfix
    rsi_feature_with_postfix = factory.create_feature("rsi", source_data, config, postfix="_14")
    assert rsi_feature_with_postfix is not None
    assert isinstance(rsi_feature_with_postfix, RsiFeature)

    # Test nonexistent feature
    nonexistent_feature = factory.create_feature("nonexistent", source_data, config)
    assert nonexistent_feature is None


def test_get_registry(factory: FeatureFactory) -> None:
    """Test that the factory returns its registry."""
    # When
    registry = factory.get_registry()

    # Then
    assert registry is not None
    assert isinstance(registry, FeatureClassRegistryInterface)


def test_get_config_registry(factory: FeatureFactory) -> None:
    """Test that the factory returns its config registry."""
    # When
    config_registry = factory.get_config_registry()

    # Then
    assert config_registry is not None
    assert isinstance(config_registry, FeatureConfigRegistryInterface)


def test_get_config_class(factory: FeatureFactory) -> None:
    """Test that the factory can retrieve config classes."""
    # When & Then
    rsi_config_class = factory.get_config_class("rsi")
    assert rsi_config_class is RsiConfig

    macd_config_class = factory.get_config_class("macd")
    assert macd_config_class is MacdConfig

    # Test nonexistent config
    nonexistent_config = factory.get_config_class("nonexistent")
    assert nonexistent_config is None


def test_create_config_instance(factory: FeatureFactory) -> None:
    """Test that the factory can create config instances."""
    # Given
    rsi_config_data = {"length": 21, "enabled": True}
    macd_config_data = {"fast": 10, "slow": 20, "signal": 5, "enabled": True}

    # When & Then
    rsi_config = factory.create_config_instance("rsi", rsi_config_data)
    assert rsi_config is not None
    assert isinstance(rsi_config, RsiConfig)
    assert rsi_config.length == 21
    assert rsi_config.enabled is True
    assert rsi_config.type == "rsi"

    macd_config = factory.create_config_instance("macd", macd_config_data)
    assert macd_config is not None
    assert isinstance(macd_config, MacdConfig)
    assert macd_config.fast == 10
    assert macd_config.slow == 20
    assert macd_config.signal == 5

    # Test nonexistent feature
    nonexistent_config = factory.create_config_instance("nonexistent", {})
    assert nonexistent_config is None
