from typing import Any, Dict

from drl_trading_strategy.feature.indicator_backend_registry import (
    IndicatorBackendRegistry,
)


class TechnicalIndicatorService:
    def __init__(self, registry: IndicatorBackendRegistry):
        self.registry = registry
        self.instances: Dict[str, Any] = {}

    def register_instance(self, name: str, indicator_type: str, **params):
        if name in self.instances:
            raise ValueError(f"Indicator instance '{name}' already exists")
        self.instances[name] = self.registry.get_indicator(indicator_type, **params)

    def update(self, name: str, value: Any):
        self.instances[name].add(value)

    def latest(self, name: str) -> Any:
        return self.instances[name][-1]
