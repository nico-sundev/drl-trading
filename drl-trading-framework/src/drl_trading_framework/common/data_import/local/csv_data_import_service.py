import os
from typing import List, Optional, Tuple

import dask
import pandas as pd
from dask import delayed
from drl_trading_common.config.local_data_import_config import LocalDataImportConfig

from drl_trading_framework.common.model.asset_price_dataset import AssetPriceDataSet
from drl_trading_framework.common.model.asset_price_import_properties import (
    AssetPriceImportProperties,
)
from drl_trading_framework.common.model.symbol_import_container import (
    SymbolImportContainer,
)

from ..base_data_import_service import BaseDataImportService


class CsvDataImportService(BaseDataImportService):
    """Service to import OHLC data from CSV files."""

    def __init__(self, config: LocalDataImportConfig):
        """
        Initializes with local data import configuration.
        It dynamically determines the project root.

        Args:
            config: Configuration containing symbols with their datasets
        """
        self.config = config
        # Determine project root dynamically.
        # Assumes this file is at <project_root>/src/drl_trading_framework/common/data_import/local/csv_data_import_service.py
        # So, project_root is 5 levels up from this file's directory.
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
        )

    def _load_csv(self, file_path: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Loads a CSV file into a DataFrame with Time as the index.

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
        Fills an AssetPriceDataSet with data from a CSV file.

        Args:
            dataset_properties (AssetPriceImportProperties): Properties of the dataset
            symbol (str): Symbol for which the dataset is being filled

        Returns:
            Tuple[str, AssetPriceDataSet]: A tuple containing the symbol and the filled AssetPriceDataSet
        """
        return symbol, AssetPriceDataSet(
            timeframe=dataset_properties.timeframe,
            base_dataset=dataset_properties.base_dataset,
            asset_price_dataset=self._load_csv(
                dataset_properties.file_path, self.config.limit
            ),
        )

    def import_data(self) -> List[SymbolImportContainer]:
        """
        Imports data from CSV files for all symbols.

        Args:
            limit: Optional limit on number of rows to read

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
