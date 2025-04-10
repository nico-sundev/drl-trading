
import math

import os
import pandas as pd
from ai_trading.config.config_loader import ConfigLoader
from ai_trading.data_import.data_import_manager import DataImportManager
from ai_trading.data_import.local.csv_data_import_service import CsvDataImportService
from ai_trading.model.preprocessed_dataset_container import PreprocessedDataSetContainer
from ai_trading.preprocess.feature.feature_aggregator import FeatureAggregator
from ai_trading.config.feature_config import FeaturesConfig
from ai_trading.preprocess.feature.feature_factory import FeatureFactory
from ai_trading.preprocess.merging_service import MergingService

class DataSetPrepService:
    
    def __init__(self, feature_config: FeaturesConfig):
        self.feature_config = feature_config
    
    # define config:
    # ticker
    # file path of datasets -> timeframes
    # train val tst ratio
    
    def get_data_sets(self) -> PreprocessedDataSetContainer:
        return self.split_time_series(self.prepare_data_set())
    
    def split_time_series(self, df: pd.DataFrame, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1) -> PreprocessedDataSetContainer:
        assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1.0"
        
        n = len(df)
        train_end = round(train_ratio * n)
        val_end = train_end + round(val_ratio * n)

        df_train = df.iloc[:train_end]
        df_val = df.iloc[train_end:val_end]
        df_test = df.iloc[val_end:]

        return PreprocessedDataSetContainer(df_train, df_val, df_test)

    
    def prepare_data_set(self):
        file_paths = {
                "H1": os.path.join(os.path.dirname(__file__), "..\\..\\data\\raw\\EURUSD_H1.csv"),
                "H4": os.path.join(os.path.dirname(__file__), "..\\..\\data\\raw\\EURUSD_H4.csv"),
            }
        repository = CsvDataImportService(file_paths)
        importer: DataImportManager = DataImportManager(repository)
        datasets = importer.get_data(1000)
        df_1h = datasets["H1"]
        df_4h = datasets["H4"]
        

        # Load Config
        config: FeaturesConfig = ConfigLoader.feature_config(os.path.join(os.path.dirname(__file__), "..\\..\\configs\\featuresConfig.json"))
        
        # Feature Engineering
        fe_1h: FeatureFactory = FeatureFactory(df_1h)
        fe_4h: FeatureFactory = FeatureFactory(df_4h)
        feat_aggregator_1h = FeatureAggregator(fe_1h, config)
        feat_aggregator_4h = FeatureAggregator(fe_4h, config)
        
        # Only test integration of one indicator, to keep it simple and quicker
        feature_df_1h = feat_aggregator_1h.compute()
        feature_df_4h = feat_aggregator_4h.compute()
        
        
        # Merge Timeframes
        merger: MergingService = MergingService(feature_df_1h, feature_df_4h)
        return merger.merge_timeframes()