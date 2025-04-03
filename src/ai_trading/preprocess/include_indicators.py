import os
import numpy as np
from IPython.display import display
import pandas as pd
from ai_trading.config.config_loader import ConfigLoader
from ai_trading.data_import.data_import_manager import DataImportManager
from ai_trading.data_import.local.csv_data_import_service import CsvDataImportService
from ai_trading.data_import.web.import_data import create_data_sets, fetch_stock_data
from ai_trading.model.dataset_container import DataSetContainer
from ai_trading.preprocess.feature.feature_aggregator import FeatureAggregator
from ai_trading.preprocess.feature.feature_config import FeatureConfig
from ai_trading.preprocess.feature.feature_factory import FeatureFactory
from ai_trading.preprocess.merging_service import MergingService

def prepare_data_set():
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
    config: FeatureConfig = ConfigLoader.from_json(os.path.join(os.path.dirname(__file__), "..\\..\\configs\\featuresConfig.json"))
    
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
    

def add_technical_indicators(df) -> pd.DataFrame:

    df = df.copy()
    return df


def preprocess_dataset(tickers) -> DataSetContainer:
    # Call the function to get data
    stock_data = fetch_stock_data(tickers, "2009-01-01", "2020-05-08")
    data_sets = create_data_sets(stock_data)
    # add technical indicators to the training data for each stock
    for ticker, df in data_sets.training_data.items():
        data_sets.training_data[ticker] = add_technical_indicators(df)

    # add technical indicators to the validation data for each stock
    for ticker, df in data_sets.validation_data.items():
        data_sets.validation_data[ticker] = add_technical_indicators(df)

    # add technical indicators to the test data for each stock
    for ticker, df in data_sets.test_data.items():
        data_sets.test_data[ticker] = add_technical_indicators(df)

    return data_sets
