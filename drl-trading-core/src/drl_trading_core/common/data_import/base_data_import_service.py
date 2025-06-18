from abc import ABC, abstractmethod
from typing import List

from drl_trading_common.config.local_data_import_config import LocalDataImportConfig

from drl_trading_core.common.model.symbol_import_container import (
    SymbolImportContainer,
)


class BaseDataImportService(ABC):
    """Abstract interface for data import services."""

    def __init__(self, config: LocalDataImportConfig):
        self.config = config

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
