from typing import Type
from unittest.mock import MagicMock

import pytest
from drl_trading_common import BaseParameterSetConfig
from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.interface.indicator.technical_indicator_facade_interface import (
    ITechnicalIndicatorFacade,
)
from drl_trading_common.model.dataset_identifier import DatasetIdentifier
from drl_trading_strategy_example.feature.feature_factory import FeatureFactory
from drl_trading_strategy_example.feature.registry.feature_class_registry_interface import (
    IFeatureClassRegistry,
)
from drl_trading_strategy_example.feature.registry.feature_config_registry_interface import (
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

    def get_feature_class(feature_type: str) -> Type[BaseFeature]:
        if feature_type.lower() in feature_mapping:
            return feature_mapping[feature_type.lower()]
        return None  # Return None for unknown feature types

    mock.get_feature_class.side_effect = get_feature_class

    # Add has_feature_class mock (case-insensitive like real implementation)
    def has_feature_class(feature_type: str) -> bool:
        return feature_type.lower() in feature_mapping

    mock.has_feature_class.side_effect = has_feature_class
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

    def get_config_class(feature_type: str) -> Type[BaseParameterSetConfig]:
        return config_mapping.get(feature_type.lower())

    mock.get_config_class.side_effect = get_config_class

    # Add has_feature_config mock
    def has_feature_config(feature_type: str) -> bool:
        return feature_type.lower() in config_mapping

    mock.has_feature_config.side_effect = has_feature_config
    mock.reset.side_effect = lambda: None
    return mock


@pytest.fixture
def factory(class_registry, config_registry, mock_indicator_service) -> FeatureFactory:
    return FeatureFactory(class_registry, config_registry, mock_indicator_service)


def test_create_feature(
    factory: FeatureFactory,
    mock_indicator_service: ITechnicalIndicatorFacade,
    mock_rsi_feature_class: Type[BaseFeature],
    mock_macd_feature_class: Type[BaseFeature],
) -> None:
    """Test that feature instances can be created using the factory."""
    # Given
    dataset_id = DatasetIdentifier(symbol="EURUSD", timeframe="H1")
    config = MockConfig()

    # When & Then
    macd_feature = factory.create_feature(
        "macd", dataset_id, config
    )
    assert macd_feature is not None
    assert isinstance(macd_feature, mock_macd_feature_class)

    rsi_feature = factory.create_feature(
        "rsi", dataset_id, config
    )
    assert rsi_feature is not None
    assert isinstance(rsi_feature, mock_rsi_feature_class)

    # Test with postfix
    rsi_feature_with_postfix = factory.create_feature(
        "rsi", dataset_id, config, postfix="_14"
    )
    assert rsi_feature_with_postfix is not None
    assert isinstance(rsi_feature_with_postfix, mock_rsi_feature_class)

    # Test nonexistent feature
    nonexistent_feature = factory.create_feature(
        "nonexistent", dataset_id, config
    )
    assert nonexistent_feature is None


def test_create_feature_without_config() -> None:
    """Test that features can be created without a config parameter."""
    # Given
    dataset_id = DatasetIdentifier(symbol="EURUSD", timeframe="H1")

    # Create a mock feature class that accepts None config
    class MockNoConfigFeature(BaseFeature):
        def __init__(self, dataset_id, indicator_service, config=None, postfix=""):
            super().__init__(dataset_id, indicator_service, config, postfix)

        def get_sub_features_names(self):
            return []

        def get_feature_name(self):
            return "close"

        def get_config_to_string(self):
            return ""

        def compute_all(self):
            return None

        def update(self, df):
            pass

        def compute_latest(self):
            return None

    # Create mock registries
    mock_class_registry = MagicMock(spec=IFeatureClassRegistry)
    mock_class_registry.get_feature_class.return_value = MockNoConfigFeature

    mock_config_registry = MagicMock(spec=IFeatureConfigRegistry)
    mock_config_registry.get_config_class.return_value = None  # No config class needed

    mock_indicator_service = MagicMock(spec=ITechnicalIndicatorFacade)

    # Create factory with mocked dependencies
    factory = FeatureFactory(mock_class_registry, mock_config_registry, mock_indicator_service)

    # Test that registry returns the class
    feature_class = factory._registry.get_feature_class("close")
    assert feature_class == MockNoConfigFeature

    # Test creating instance directly
    test_instance = MockNoConfigFeature(dataset_id, mock_indicator_service, config=None)
    assert test_instance is not None

    # When & Then - Test with None config explicitly
    feature_with_none = factory.create_feature(
        "close", dataset_id, config=None
    )
    assert feature_with_none is not None
    assert isinstance(feature_with_none, MockNoConfigFeature)


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


def test_is_feature_supported_returns_true_for_supported_features(factory: FeatureFactory) -> None:
    """Test that is_feature_supported returns True for features with registered classes."""
    # Given & When & Then
    assert factory.is_feature_supported("rsi") is True
    assert factory.is_feature_supported("macd") is True
    assert factory.is_feature_supported("test") is False  # Has config but no feature class


def test_is_feature_supported_returns_false_for_unsupported_features(factory: FeatureFactory) -> None:
    """Test that is_feature_supported returns False for features without registered classes."""
    # Given & When & Then
    assert factory.is_feature_supported("nonexistent") is False
    assert factory.is_feature_supported("unknown_feature") is False
    assert factory.is_feature_supported("") is False


def test_is_feature_supported_handles_case_insensitive_lookup(factory: FeatureFactory) -> None:
    """Test that is_feature_supported works with case-insensitive feature names."""
    # Given & When & Then
    assert factory.is_feature_supported("RSI") is True
    assert factory.is_feature_supported("rsi") is True
    assert factory.is_feature_supported("RsI") is True
    assert factory.is_feature_supported("MACD") is True
    assert factory.is_feature_supported("macd") is True


def test_is_feature_supported_handles_registry_exceptions(class_registry, config_registry, mock_indicator_service) -> None:
    """Test that is_feature_supported handles registry exceptions gracefully."""
    # Given
    class_registry.has_feature_class.side_effect = Exception("Registry error")
    factory = FeatureFactory(class_registry, config_registry, mock_indicator_service)

    # When & Then
    assert factory.is_feature_supported("rsi") is False


def test_is_feature_supported_logs_validation_details(factory: FeatureFactory, caplog) -> None:
    """Test that is_feature_supported provides detailed logging for debugging."""
    # Given
    import logging
    caplog.set_level(logging.DEBUG)

    # When
    result_supported = factory.is_feature_supported("rsi")
    result_unsupported = factory.is_feature_supported("nonexistent")

    # Then
    assert result_supported is True
    assert result_unsupported is False

    # Check debug logs contain validation details
    debug_messages = [record.message for record in caplog.records if record.levelname == 'DEBUG']
    assert any("class=True, config=True" in msg for msg in debug_messages)
    assert any("Feature class not found for 'nonexistent'" in msg for msg in debug_messages)
