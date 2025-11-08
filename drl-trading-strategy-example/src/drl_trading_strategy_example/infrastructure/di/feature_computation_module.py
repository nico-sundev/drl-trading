"""Feature computation DI module - provides features, indicators, and registries.

Used by preprocess, inference, and training services for feature computation.
For training, combine with RLEnvironmentModule to get gym environment.
"""

import logging

from injector import Module, provider, singleton

from drl_trading_common.interface.feature.feature_factory_interface import IFeatureFactory
from drl_trading_common.interface.indicator.technical_indicator_facade_interface import (
    ITechnicalIndicatorFacade,
)
from drl_trading_strategy_example.feature.feature_factory import FeatureFactory
from drl_trading_strategy_example.feature.registry.feature_class_registry import (
    FeatureClassRegistry,
)
from drl_trading_strategy_example.feature.registry.feature_config_registry import (
    FeatureConfigRegistry,
)
from drl_trading_strategy_example.feature.registry.feature_class_registry_interface import (
    IFeatureClassRegistry,
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

logger = logging.getLogger(__name__)


class FeatureComputationModule(Module):
    """DI module for feature computation (features + indicators, no RL environment)."""

    def configure(self, binder) -> None:  # type: ignore[no-untyped-def,override]
        """Configure interface bindings."""
        binder.bind(IFeatureFactory, to=FeatureFactory, scope=singleton)  # type: ignore[type-abstract]
        binder.bind(ITechnicalIndicatorFacade, to=TaLippIndicatorService, scope=singleton)  # type: ignore[type-abstract]
        logger.info("FeatureComputationModule configured")

    @provider
    @singleton
    def provide_feature_class_registry(self) -> IFeatureClassRegistry:
        """Provide feature class registry with auto-discovery."""
        registry = FeatureClassRegistry()
        registry.discover_feature_classes("drl_trading_strategy_example.feature.collection.observable")
        registry.discover_feature_classes("drl_trading_strategy_example.feature.collection.context_related")
        logger.info(f"FeatureClassRegistry: {len(registry)} features")
        return registry

    @provider
    @singleton
    def provide_feature_config_registry(self) -> IFeatureConfigRegistry:
        """Provide feature config registry with auto-discovery."""
        registry = FeatureConfigRegistry()
        registry.discover_config_classes("drl_trading_strategy_example.feature.config")
        logger.info(f"FeatureConfigRegistry: {len(registry)} configs")
        return registry

    @provider
    @singleton
    def provide_indicator_class_registry(self) -> IndicatorClassRegistryInterface:
        """Provide indicator registry with auto-discovery."""
        registry = IndicatorClassRegistry()
        registry.discover_indicator_classes("drl_trading_strategy_example.technical_indicator.collection")
        logger.info(f"IndicatorClassRegistry: {len(registry)} indicators")
        return registry
