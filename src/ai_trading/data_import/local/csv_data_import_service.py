from typing import List, Optional
import dask
from dask import delayed
import pandas as pd
from ai_trading.model.asset_price_dataset import AssetPriceDataSet
from ai_trading.model.asset_price_import_properties import AssetPriceImportProperties
from ..base_data_import_service import BaseDataImportService


class CsvDataImportService(BaseDataImportService):
    """Service to import OHLC data from CSV files."""

    def __init__(self, importProperties: List[AssetPriceImportProperties]):
        """
        Initializes with file paths.

        :param file_paths: Dictionary where keys are timeframes (e.g., "H1", "H4") and values are file paths.
        """
        self.importProperties = importProperties

    def _load_csv(self, file_path: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Loads a CSV file into a DataFrame."""
        return pd.read_csv(
            file_path,
            usecols=["Time", "Open", "High", "Low", "Close"],
            sep="\t",
            parse_dates=["Time"],
            skipinitialspace=True,
            nrows=limit,
        )

    def import_data(self, limit: Optional[int] = None) -> List[AssetPriceDataSet]:
        """Imports data from CSV files."""

        compute_tasks = [
            delayed(self._load_csv)(dataset_property.file_path, limit)  # Delay the _load_csv method
            for dataset_property in self.importProperties
        ]
        
        # Now for each dataset, construct the AssetPriceDataSet
        asset_price_datasets = [
            AssetPriceDataSet(dataset_property.timeframe, dataset_property.base_dataset, data)
            for dataset_property, data in zip(self.importProperties, dask.compute(*compute_tasks))  # Compute the results of the delayed tasks
        ]
        
        return asset_price_datasets
