import os
import pytest
from ai_trading.config.config_loader import ConfigLoader
from ai_trading.config.feature_config_registry import FeatureConfigRegistry
from ai_trading.data_import.data_import_manager import DataImportManager
from ai_trading.data_import.local.csv_data_import_service import CsvDataImportService
from ai_trading.preprocess.feature.feature_aggregator import FeatureAggregator
from ai_trading.preprocess.feature.feature_class_registry import FeatureClassRegistry


@pytest.fixture
def dataset():
    file_paths = {
            "H1": os.path.join(os.path.dirname(__file__), "../resources/test_ohlc_dataset.csv")
        }
    repository = CsvDataImportService(file_paths)
    importer = DataImportManager(repository)
    return importer.get_data()["H1"]

@pytest.fixture(autouse=True)
def reset_registry():
    FeatureConfigRegistry._instance = None

@pytest.fixture(autouse=True)
def config(reset_registry):
    return ConfigLoader.get_config(os.path.join(os.path.dirname(__file__), "../resources/applicationConfig-test.json"))

@pytest.fixture(autouse=True)
def class_registry():
    return FeatureClassRegistry()

@pytest.fixture
def feature_aggregator(dataset, config, class_registry):
    return FeatureAggregator(dataset, config.features_config, class_registry)

def test_features(feature_aggregator: FeatureAggregator):
    # Given
    expected_columns = ['Time', 'rsi_7']
    
    # When
    result_df = feature_aggregator.compute()
    
    # Then
    assert set(expected_columns).issubset(set(result_df.columns))
