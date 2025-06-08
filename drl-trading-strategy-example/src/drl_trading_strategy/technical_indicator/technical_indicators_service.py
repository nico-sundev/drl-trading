from typing import Any, Dict, Optional

from drl_trading_common.base.base_indicator import BaseIndicator
from drl_trading_common.interfaces.indicator.technical_indicator_facade_interface import (
    TechnicalIndicatorFacadeInterface,
    TechnicalIndicatorFactoryInterface,
)
from drl_trading_strategy.enum.indicator_type_enum import IndicatorTypeEnum
from drl_trading_strategy.technical_indicator.indicator_class_registry import (
    IndicatorClassRegistry,
)
from injector import inject
from pandas import DataFrame


class TaLippIndicatorService(TechnicalIndicatorFacadeInterface):

    @inject
    def __init__(self, registry: IndicatorClassRegistry) -> None:
        self.instances: Dict[str, BaseIndicator] = {}
        self.registry = registry

    def register_instance(self, name: str, indicator_type: IndicatorTypeEnum, **params) -> None:
        if name in self.instances:
            raise ValueError(f"Indicator with name {name} already exists.")
        indicator_class = self.registry.get_indicator_class(indicator_type)

        if not indicator_class:
            raise ValueError(f"No indicator class found for type {indicator_type}")
        self.instances[name] = indicator_class(**params)

    def get_all(self, name: str) -> Optional[DataFrame]:
        return self.instances[name].get_all()

    def add(self, name: str, values: DataFrame) -> None:
        self.instances[name].add(values)

    def get_latest(self, name: str) -> Optional[DataFrame]:
        return self.instances[name].get_latest()

class TaLippIndicatorFactory(TechnicalIndicatorFactoryInterface):

    def create(self, **kwargs: Any) -> TechnicalIndicatorFacadeInterface:
        """
        Factory method to create an instance of the technical indicator service.

        Returns:
            An instance of the technical indicator service.
        """
        return TaLippIndicatorService(IndicatorClassRegistry())
