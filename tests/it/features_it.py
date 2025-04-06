import os
from pandas import DataFrame, concat
import pytest
from ai_trading.config.config_loader import ConfigLoader
from ai_trading.data_import.data_import_manager import DataImportManager
from ai_trading.data_import.local.csv_data_import_service import CsvDataImportService
from ai_trading.preprocess.feature.collection.feature_mapper import FEATURE_MAP
from ai_trading.preprocess.feature.feature_aggregator import FeatureAggregator


@pytest.fixture
def dataset():
    file_paths = {
            "H1": os.path.join(os.path.dirname(__file__), "../resources/test_ohlc_dataset.csv")
        }
    repository = CsvDataImportService(file_paths)
    importer = DataImportManager(repository)
    return importer.get_data()["H1"]

@pytest.fixture
def config():
    return ConfigLoader.get_config(os.path.join(os.path.dirname(__file__), "../resources/applicationConfig-test.json"))

@pytest.fixture
def feature_aggregator(dataset, config):
    return FeatureAggregator(dataset, config.features_config)

def test_features(feature_aggregator: FeatureAggregator):
    # Given
    expected_columns = ['Time', 'rsi_7', 'rsi_14', 'rsi_21']
    
    # When
    result_df = feature_aggregator.compute()
    
    # Then
    assert set(result_df.columns) == set(expected_columns)
