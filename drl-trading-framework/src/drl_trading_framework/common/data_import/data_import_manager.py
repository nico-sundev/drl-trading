from typing import List

from injector import inject

from drl_trading_framework.common.data_import.base_data_import_service import (
    BaseDataImportService,
)
from drl_trading_framework.common.model.symbol_import_container import (
    SymbolImportContainer,
)


class DataImportManager:
    """Manages data import from different sources."""

    @inject
    def __init__(self, import_service: BaseDataImportService):
        """
        Initializes with a data import service.

        Args:
            import_service: Instance of a class implementing BaseDataImportService.
        """
        self.import_service = import_service

    def get_data(self) -> List[SymbolImportContainer]:
        """
        Fetches OHLC data using the selected service.

        Args:
            limit: Optional limit on number of rows to read

        Returns:
            List of SymbolImportContainer objects, one for each symbol
        """
        return self.import_service.import_data()
