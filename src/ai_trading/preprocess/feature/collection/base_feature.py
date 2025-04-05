
from abc import ABC, abstractmethod

from pandas import DataFrame

from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig


class BaseFeature(ABC):
    
    @abstractmethod
    def compute(self, config: BaseParameterSetConfig) -> DataFrame:
        pass