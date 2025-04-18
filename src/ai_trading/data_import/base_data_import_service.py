from abc import ABC, abstractmethod
from typing import List, Optional

from ai_trading.model.asset_price_dataset import AssetPriceDataSet


class BaseDataImportService(ABC):
    """Abstract interface for data import services."""

    @abstractmethod
    def import_data(self, limit: Optional[int] = None) -> List[AssetPriceDataSet]:
        """Imports data and returns a dictionary of DataFrames."""
        pass
