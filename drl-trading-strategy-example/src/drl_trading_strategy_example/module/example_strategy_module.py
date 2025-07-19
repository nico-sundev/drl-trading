"""
Bootstrap module for the drl-trading-strategy-example package.

This module provides concrete implementations and DI configuration
for the trading framework.
"""

from typing import Type

from drl_trading_common.base.base_strategy_module import BaseStrategyModule
from drl_trading_common.base.base_trading_env import BaseTradingEnv
from drl_trading_common.interface.feature.context_feature_service_interface import (
    ContextFeatureServiceInterface,
)
from drl_trading_common.interface.indicator.technical_indicator_facade_interface import (
    ITechnicalIndicatorFacade,
)
from injector import Module, provider, singleton

from drl_trading_strategy_example.feature.context.context_feature_service import (
    ContextFeatureService,
)
from drl_trading_strategy_example.feature.registry.feature_class_registry import (
    FeatureClassRegistry,
)
from drl_trading_strategy_example.feature.registry.feature_class_registry_interface import (
    IFeatureClassRegistry,
)
from drl_trading_strategy_example.feature.registry.feature_config_registry import (
    FeatureConfigRegistry,
)
from drl_trading_strategy_example.feature.registry.feature_config_registry_interface import (
    IFeatureConfigRegistry,
)
from drl_trading_strategy_example.technical_indicator.registry.indicator_class_registry import (
    IndicatorClassRegistry,
)
from drl_trading_strategy_example.technical_indicator.registry.indicator_class_registry_interface import (
    IndicatorClassRegistryInterface,
)
from drl_trading_strategy_example.technical_indicator.talipp_indicator_service import (
    TaLippIndicatorService,
)

from ..gym_env.custom_env import MyCustomTradingEnv


class ExampleStrategyModule(BaseStrategyModule):

    def as_injector_module(self) -> Module:
        """
        DI module providing concrete implementations for the trading framework.

        This module provides the concrete registry implementation that will be
        used by the framework's factory.
        """

        class _Internal(Module):

            @provider
            @singleton
            def provide_feature_class_registry(self) -> IFeatureClassRegistry:
                """Provide the concrete feature class registry implementation."""
                registry = FeatureClassRegistry()
                # Discover features from the impl package
                registry.discover_feature_classes(
                    "drl_trading_strategy.feature.collection"
                )
                return registry

            @provider
            @singleton
            def provide_feature_config_registry(self) -> IFeatureConfigRegistry:
                """Provide the concrete feature config registry implementation."""
                registry = FeatureConfigRegistry()
                # Discover config classes from the impl package
                registry.discover_config_classes(
                    "drl_trading_strategy.feature.collection"
                )
                return registry

            @provider
            @singleton
            def provide_indicator_class_registry(self) -> IndicatorClassRegistryInterface:
                registry = IndicatorClassRegistry()
                registry.discover_indicator_classes(
                    "drl_trading_strategy.technical_indicator.collection"
                )
                return registry

            @provider
            @singleton
            def provide_trading_environment_class(self) -> Type[BaseTradingEnv]:
                """Provide the custom trading environment class."""
                return MyCustomTradingEnv

            @provider
            @singleton
            def provide_technical_indicator_facade(
                self,
                registry: IndicatorClassRegistryInterface,
            ) -> ITechnicalIndicatorFacade:
                """Provide the indicator backend registry implementation."""
                return TaLippIndicatorService(registry)

            @provider
            @singleton
            def provide_context_feature_service(
                self,
            ) -> ContextFeatureServiceInterface:
                """Provide the context feature service."""
                svc = ContextFeatureService()
                return svc

        return _Internal()
