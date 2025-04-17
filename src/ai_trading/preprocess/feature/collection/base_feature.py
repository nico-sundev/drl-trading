
from abc import ABC, abstractmethod

from pandas import DataFrame

from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig


class BaseFeature(ABC):
    
    @abstractmethod
    def compute(self, config: BaseParameterSetConfig) -> DataFrame:
        pass
    
    @abstractmethod
    def get_sub_features_names(self, config: BaseParameterSetConfig) -> list[str]:
        """ Get the names of the sub-features.
        This method should be implemented by subclasses to return the names of the sub-features.

        Returns:
            list[str]: A list of sub-feature names.
        """
        pass