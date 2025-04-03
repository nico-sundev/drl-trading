import os
import pandas as pd

from ai_trading.config.config_loader import ConfigLoader
from ai_trading.data_import.data_import_manager import DataImportManager
from ai_trading.data_import.local.csv_data_import_service import CsvDataImportService
from ai_trading.preprocess.feature.feature_config import FeatureConfig
from ai_trading.preprocess.feature.feature_factory import FeatureFactory
from ai_trading.preprocess.merging_service import MergingService


def test_preprocessing():
        
    file_paths = {
            "H1": os.path.join(os.path.dirname(__file__), "..\\..\\data\\raw\\EURUSD_H1.csv"),
            "H4": os.path.join(os.path.dirname(__file__), "..\\..\\data\\raw\\EURUSD_H4.csv"),
        }
    repository = CsvDataImportService(file_paths)
    importer: DataImportManager = DataImportManager(repository)
    datasets = importer.get_data(100)
    df_1h = datasets["H1"]
    df_4h = datasets["H4"]
    

    # Load Config
    config: FeatureConfig = ConfigLoader.from_json(os.path.join(os.path.dirname(__file__), "..\\..\\configs\\featuresConfig.json"))
    
    # Feature Engineering
    fe_1h: FeatureFactory = FeatureFactory(df_1h)
    fe_4h: FeatureFactory = FeatureFactory(df_4h)
    
    # Only test integration of one indicator, to keep it simple and quicker
    feature_df_1h = fe_1h.compute_rsi(config.rsi_lengths[0])
    feature_df_4h = fe_4h.compute_rsi(config.rsi_lengths[0])
    
    # Merge Timeframes
    merger: MergingService = MergingService(feature_df_1h, feature_df_4h)
    feature_df_merged: pd.DataFrame = merger.merge_timeframes()

    expected_columns = {"Time", f"rsi_{config.rsi_lengths[0]}", f"HTF240_rsi_{config.rsi_lengths[0]}"}  
    actual_columns = set(feature_df_merged.columns)

    assert actual_columns == expected_columns, f"Column mismatch! Expected: {expected_columns}, but got: {actual_columns}"

    #print(feature_df_merged.head())
