import pandas as pd
from typing import Dict, Optional
from ..base_data_import_service import BaseDataImportService


class CsvDataImportService(BaseDataImportService):
    """Service to import OHLC data from CSV files."""

    def __init__(self, file_paths: Dict[str, str]):
        """
        Initializes with file paths.

        :param file_paths: Dictionary where keys are timeframes (e.g., "H1", "H4") and values are file paths.
        """
        self.file_paths = file_paths

    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Loads a CSV file into a DataFrame."""
        return pd.read_csv(
            file_path,
            usecols=["Time", "Open", "High", "Low", "Close"],
            sep="\t",
            parse_dates=["Time"],
            skipinitialspace=True,
        )

    def import_data(self, limit: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Imports data from CSV files."""
        data = {}
        for timeframe, path in self.file_paths.items():
            df = self._load_csv(path)
            data[timeframe] = df.head(limit) if limit is not None else df
        return data
