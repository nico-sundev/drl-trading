from abc import ABC, abstractmethod
from typing import Optional, Type

from drl_trading_common.base.base_indicator import BaseIndicator
from drl_trading_strategy.enum.indicator_type_enum import IndicatorTypeEnum


class IndicatorClassRegistryInterface(ABC):

    @abstractmethod
    def get_indicator_class(self, indicator_type: IndicatorTypeEnum) -> Optional[Type[BaseIndicator]]:
        pass

    @abstractmethod
    def register_indicator_class(
        self, indicator_type: IndicatorTypeEnum, indicator_class: Type[BaseIndicator]
    ) -> None:
        pass
