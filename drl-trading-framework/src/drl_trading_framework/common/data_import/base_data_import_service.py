from abc import ABC, abstractmethod
from typing import List

from drl_trading_framework.common.model.symbol_import_container import (
    SymbolImportContainer,
)


class BaseDataImportService(ABC):
    """Abstract interface for data import services."""

    @abstractmethod
    def import_data(self) -> List[SymbolImportContainer]:
        """
        Imports data from CSV files for all symbols.

        Args:
            limit: Optional limit on number of rows to read

        Returns:
            List of SymbolImportContainer objects, one for each symbol
        """
        pass
