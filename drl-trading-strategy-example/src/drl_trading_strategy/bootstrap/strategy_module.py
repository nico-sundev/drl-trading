"""
Bootstrap module for the drl-trading-strategy-example package.

This module provides concrete implementations and DI configuration
for the trading framework.
"""
from typing import Type

from drl_trading_common.base.base_trading_env import BaseTradingEnv
from drl_trading_common.interfaces.feature.feature_class_registry_interface import (
    FeatureClassRegistryInterface,
)
from drl_trading_common.interfaces.feature.feature_config_registry_interface import (
    FeatureConfigRegistryInterface,
)
from injector import Module, provider, singleton

from ..custom_env import MyCustomTradingEnv
from ..feature.feature_class_registry import FeatureClassRegistry
from ..feature.feature_config_registry import FeatureConfigRegistry


class StrategyModule(Module):
    """
    DI module providing concrete implementations for the trading framework.

    This module provides the concrete registry implementation that will be
    used by the framework's factory.
    """

    @provider
    @singleton
    def provide_feature_class_registry(self) -> FeatureClassRegistryInterface:
        """Provide the concrete feature class registry implementation."""
        registry = FeatureClassRegistry()
        # Discover features from the impl package
        registry.discover_feature_classes("drl_trading_strategy.feature.collection")
        return registry

    @provider
    @singleton
    def provide_feature_config_registry(self) -> FeatureConfigRegistryInterface:
        """Provide the concrete feature config registry implementation."""
        registry = FeatureConfigRegistry()
        # Discover config classes from the impl package
        registry.discover_config_classes("drl_trading_strategy.feature.collection")
        return registry

    @provider
    @singleton
    def provide_trading_environment_class(self) -> Type[BaseTradingEnv]:
        """Provide the custom trading environment class."""
        return MyCustomTradingEnv
