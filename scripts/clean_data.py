import os

from ai_trading.data_import.data_import_manager import DataImportManager
from ai_trading.data_import.local.csv_data_import_service import CsvDataImportService

ticker = "EURUSD"
file_paths = {
    "H1": os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "raw", f"{ticker}_H1.csv")
    ),
    "H4": os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "raw", f"{ticker}_H4.csv")
    ),
}
repository = CsvDataImportService(file_paths)
importer = DataImportManager(repository)
datasets = importer.get_data()

for dfkey in datasets.keys():
    datasets[dfkey].to_parquet(
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "processed",
                f"{ticker}_{dfkey}.parquet",
            )
        )
    )
