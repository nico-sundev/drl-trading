
from abc import ABC, abstractmethod

from pandas import DataFrame


class BaseFeature(ABC):
    
    @abstractmethod
    def compute() -> DataFrame:
        pass