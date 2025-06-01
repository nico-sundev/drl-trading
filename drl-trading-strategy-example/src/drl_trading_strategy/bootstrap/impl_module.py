"""
Bootstrap module for the drl-trading-strategy-example package.

This module provides concrete implementations and DI configuration
for the trading framework.
"""
from drl_trading_common.interfaces.feature.feature_class_registry_interface import (
    FeatureClassRegistryInterface,
)
from injector import Module, provider, singleton

from ..feature.feature_class_registry import FeatureClassRegistry


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
