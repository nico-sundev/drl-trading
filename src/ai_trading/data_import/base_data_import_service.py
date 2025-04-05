from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional


class BaseDataImportService(ABC):
    """Abstract interface for data import services."""

    @abstractmethod
    def import_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Imports data and returns a dictionary of DataFrames."""
        pass
