"""Service to import OHLC data from CSV files."""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import dask
import pandas as pd
from dask import delayed

from drl_trading_core.common.model.asset_price_dataset import AssetPriceDataSet
from drl_trading_core.common.model.asset_price_import_properties import (
    AssetPriceImportProperties,
)
from drl_trading_core.common.model.symbol_import_container import (
    SymbolImportContainer,
)
from drl_trading_ingest.core.port import DataProviderPort

logger = logging.getLogger(__name__)


class CsvDataImportService(DataProviderPort):
    """Service to import OHLC data from CSV files."""

    PROVIDER_NAME = "csv"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with local data import configuration.

        Args:
            config: Configuration containing file paths and symbols
        """
        super().__init__(config)
        self.local_config = config

        # Determine project root dynamically
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..")
        )

    @property
    def provider_name(self) -> str:
        """Return the unique name of this data provider."""
        return self.PROVIDER_NAME

    def setup(self) -> None:
        """Initialize CSV file access and validate paths."""
        logger.info(f"Setting up {self.provider_name} provider...")

        # Validate that base path exists
        if hasattr(self.config, 'get'):
            base_path = self.config.get('base_path', 'data/csv')
        else:
            base_path = 'data/csv'

        full_path = os.path.join(self.project_root, base_path)
        if not os.path.exists(full_path):
            logger.warning(f"CSV base path does not exist: {full_path}")

        self._is_initialized = True
        logger.info(f"{self.provider_name} provider ready")

    def teardown(self) -> None:
        """Clean up CSV resources (no-op for file-based provider)."""
        logger.info(f"Tearing down {self.provider_name} provider...")
        self._is_initialized = False

    def _load_csv(self, file_path: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load a CSV file into a DataFrame with Time as the index.

        Args:
            file_path: Path to the CSV file (can be relative to project root)
            limit: Optional limit on number of rows to read

        Returns:
            DataFrame with OHLC data and Time as DatetimeIndex
        """
        if not os.path.isabs(file_path):
            abs_file_path = os.path.join(self.project_root, file_path)
        else:
            abs_file_path = file_path

        df = pd.read_csv(
            abs_file_path,
            usecols=["Time", "Open", "High", "Low", "Close", "Volume"],
            sep="\t",
            parse_dates=["Time"],
            skipinitialspace=True,
            nrows=limit,
        )
        # Set Time as the index immediately after loading
        return df.set_index("Time")

    def _get_symbol_asset_price_tuple(
        self, dataset_properties: AssetPriceImportProperties, symbol: str
    ) -> Tuple[str, AssetPriceDataSet]:
        """
        Fill an AssetPriceDataSet with data from a CSV file.

        Args:
            dataset_properties: Properties of the dataset
            symbol: Symbol for which the dataset is being filled

        Returns:
            Tuple containing the symbol and the filled AssetPriceDataSet
        """
        return symbol, AssetPriceDataSet(
            timeframe=dataset_properties.timeframe,
            base_dataset=dataset_properties.base_dataset,
            asset_price_dataset=self._load_csv(
                dataset_properties.file_path, self.config.limit
            ),
        )

    def fetch_historical_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SymbolImportContainer]:
        """Fetch historical data from CSV files."""
        return self.import_data()

    def map_symbol(self, internal_symbol: str) -> str:
        """CSV provider uses symbols as-is (file names)."""
        return internal_symbol

    def start_streaming(
        self,
        symbols: List[str],
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        """CSV provider does not support streaming."""
        logger.warning(f"{self.provider_name} provider does not support streaming")

    def stop_streaming(self) -> None:
        """CSV provider does not support streaming."""
        pass

    def import_data(self) -> List[SymbolImportContainer]:
        """
        Import data from CSV files for all symbols.

        Returns:
            List of SymbolImportContainer objects, one for each symbol
        """
        # Use provided limit or fallback to config limit
        symbol_containers = []

        for symbol_config in self.config.symbols:
            # Create tasks for all datasets of all symbols
            compute_tasks = [
                delayed(self._get_symbol_asset_price_tuple)(
                    dataset_property, symbol_config.symbol
                )
                for dataset_property in symbol_config.datasets
            ]

            # Process all datasets in parallel with dask
            computed_results = dask.compute(*compute_tasks)

            # Group datasets by symbol
            symbol_datasets: dict[str, list[AssetPriceDataSet]] = {}
            for symbol, dataset in computed_results:
                if symbol not in symbol_datasets:
                    symbol_datasets[symbol] = []
                symbol_datasets[symbol].append(dataset)

            # Create containers for each symbol
            for symbol, datasets in symbol_datasets.items():
                symbol_container = SymbolImportContainer(
                    symbol=symbol, datasets=datasets
                )
                symbol_containers.append(symbol_container)

        return symbol_containers
