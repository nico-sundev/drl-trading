import os
from typing import List

from drl_trading_common.config import LocalDataImportConfig, SymbolConfig
from drl_trading_common.models import AssetPriceImportProperties

from drl_trading_framework.common.data_import.data_import_manager import (
    DataImportManager,
)
from drl_trading_framework.common.data_import.local.csv_data_import_service import (
    CsvDataImportService,
)

ticker = "EURUSD"

# Create import properties for each timeframe
import_properties: List[AssetPriceImportProperties] = [
    AssetPriceImportProperties(
        timeframe="H1",
        base_dataset=True,
        file_path=os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "data", "raw", f"{ticker}_H1.csv"
            )
        ),
    ),
    AssetPriceImportProperties(
        timeframe="H4",
        base_dataset=False,
        file_path=os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "data", "raw", f"{ticker}_H4.csv"
            )
        ),
    ),
]

# Create a symbol config
symbol_config = SymbolConfig(symbol=ticker, datasets=import_properties)

# Create a local data import config
local_config = LocalDataImportConfig(symbols=[symbol_config])

# Initialize service with config
repository = CsvDataImportService(local_config)
importer = DataImportManager(repository)
symbol_containers = importer.get_data()

# Save datasets to parquet files
for symbol_container in symbol_containers:
    for dataset in symbol_container.datasets:
        timeframe = dataset.timeframe
        dataset.asset_price_dataset.to_parquet(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "data",
                    "processed",
                    f"{symbol_container.symbol}_{timeframe}.parquet",
                )
            )
        )
