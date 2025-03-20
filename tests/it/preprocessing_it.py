# Sample 30m Data
import os
import pandas as pd

from ai_trading.data_import.data_import_manager import DataImportManager
from ai_trading.data_import.local.csv_data_import_service import CsvDataImportService
from ai_trading.preprocess.feature_engineering.feature_engineering_service import FeatureEngineeringService
from ai_trading.preprocess.merging_service import MergingService


# Load and process data
file_paths = {
        "H1": os.path.join(os.path.dirname(__file__), "..\\..\\data\\raw\\EURUSD_H1.csv"),
        "H4": os.path.join(os.path.dirname(__file__), "..\\..\\data\\raw\\EURUSD_H4.csv"),
    }
repository = CsvDataImportService()
importer: DataImportManager = DataImportManager(repository)
datasets = importer.get_data(100)
df_1h = datasets["H1"]
df_4h = datasets["H4"]

# Feature Engineering
fe: FeatureEngineeringService = FeatureEngineeringService()
df_1h = fe.compute_features(df_1h)
df_4h = fe.compute_features(df_4h)

# Merge Timeframes
merger: MergingService = MergingService(df_1h, df_4h)
df_merged: pd.DataFrame = merger.merge_timeframes()

print(df_merged.head())
