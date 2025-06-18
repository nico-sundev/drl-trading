from typing import List

from drl_trading_common.config.local_data_import_config import LocalDataImportConfig
from injector import inject

from drl_trading_core.common.data_import.data_import_strategy_factory import (
    DataImportStrategyFactory,
)
from drl_trading_core.common.model.symbol_import_container import (
    SymbolImportContainer,
)


@inject
class DataImportManager:
    """Manages data import from different sources."""

    def __init__(
        self,
        local_data_import_config: LocalDataImportConfig,
        import_service_factory: DataImportStrategyFactory,
    ):
        """
        Initializes with a data import service.

        Args:
            import_service: Instance of a class implementing BaseDataImportService.
        """
        self.local_data_import_config = local_data_import_config
        self.import_service_factory = import_service_factory

    def get_data(self) -> List[SymbolImportContainer]:
        """
        Fetches OHLC data using the selected service.

        Args:
            limit: Optional limit on number of rows to read

        Returns:
            List of SymbolImportContainer objects, one for each symbol
        """
        import_strategy_class_instance = self.import_service_factory.create_import_service(
            self.local_data_import_config
        )
        return import_strategy_class_instance.import_data()
