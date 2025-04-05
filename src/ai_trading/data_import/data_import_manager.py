from typing import Dict, Optional
import pandas as pd

from ai_trading.data_import.base_data_import_service import BaseDataImportService


class DataImportManager:
    """Manages data import from different sources."""

    def __init__(self, import_service: BaseDataImportService):
        """
        Initializes with a data import service.

        :param import_service: Instance of a class implementing DataImportService.
        """
        self.import_service = import_service

    def get_data(self, limit: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Fetches OHLC data using the selected service."""
        return self.import_service.import_data(limit)
