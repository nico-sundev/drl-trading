from typing import Type
from unittest.mock import MagicMock

import pandas as pd
import pytest
from drl_trading_common import BaseParameterSetConfig
from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.interface.indicator.technical_indicator_facade_interface import (
    ITechnicalIndicatorFacade,
)
from drl_trading_strategy.feature.feature_factory import FeatureFactory
from drl_trading_strategy.feature.registry.feature_class_registry_interface import (
    IFeatureClassRegistry,
)
from drl_trading_strategy.feature.registry.feature_config_registry_interface import (
    IFeatureConfigRegistry,
)


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
def class_registry(
    mock_rsi_feature_class, mock_macd_feature_class
) -> IFeatureClassRegistry:
    """Create a mock feature class registry that returns mock feature classes."""
    mock = MagicMock(
        spec=IFeatureClassRegistry
    )  # Define mapping of feature types to feature classes
    feature_mapping = {
        "rsi": mock_rsi_feature_class,
        "macd": mock_macd_feature_class,
    }

    def get_feature_class(feature_type: str):
        if feature_type in feature_mapping:
            return feature_mapping[feature_type]
        return None  # Return None for unknown feature types

    mock.get_feature_class.side_effect = get_feature_class
    mock.reset.side_effect = lambda: None  # Reset method does nothing
    return mock


@pytest.fixture
def config_registry() -> IFeatureConfigRegistry:
    """Create a mock feature config registry."""
    mock = MagicMock(spec=IFeatureConfigRegistry)
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


def test_create_feature(
    factory: FeatureFactory,
    mock_indicator_service: ITechnicalIndicatorFacade,
    mock_rsi_feature_class: Type[BaseFeature],
    mock_macd_feature_class: Type[BaseFeature],
) -> None:
    """Test that feature instances can be created using the factory."""  # Given
    source_data = pd.DataFrame({"Close": [1, 2, 3, 4, 5]})
    config = MockConfig()  # When & Then
    macd_feature = factory.create_feature(
        "macd", source_data, config, mock_indicator_service
    )
    assert macd_feature is not None
    assert isinstance(macd_feature, mock_macd_feature_class)

    rsi_feature = factory.create_feature(
        "rsi", source_data, config, mock_indicator_service
    )
    assert rsi_feature is not None
    assert isinstance(rsi_feature, mock_rsi_feature_class)

    # Test with postfix
    rsi_feature_with_postfix = factory.create_feature(
        "rsi", source_data, config, mock_indicator_service, postfix="_14"
    )
    assert rsi_feature_with_postfix is not None
    assert isinstance(rsi_feature_with_postfix, mock_rsi_feature_class)

    # Test nonexistent feature
    nonexistent_feature = factory.create_feature(
        "nonexistent", source_data, config, mock_indicator_service
    )
    assert nonexistent_feature is None


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
