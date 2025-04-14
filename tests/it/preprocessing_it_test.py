import os
import pytest

from ai_trading.config.config_loader import ConfigLoader
from ai_trading.config.feature_config_registry import FeatureConfigRegistry
from ai_trading.data_import.data_import_manager import DataImportManager
from ai_trading.data_import.local.csv_data_import_service import CsvDataImportService
from ai_trading.preprocess.feature.feature_class_registry import FeatureClassRegistry
from ai_trading.preprocess.preprocess_service import PreprocessService

@pytest.fixture
def reset_registry():
    FeatureConfigRegistry._instance = None

@pytest.fixture
def config(reset_registry):
    config = ConfigLoader.get_config(
        os.path.join(
            os.path.dirname(__file__), "../resources/applicationConfig-test.json"
        )
    )
    # To speed up the test, only use RSI
    config.features_config.feature_definitions = [
        f for f in config.features_config.feature_definitions if f.name == "rsi"
    ]
    return config

@pytest.fixture
def datasets(config):
    repository = CsvDataImportService(config.local_data_import_config.datasets)
    importer = DataImportManager(repository)
    return importer.get_data(100)

@pytest.fixture
def class_registry():
    return FeatureClassRegistry()


@pytest.fixture
def preprocess_service(datasets, config, class_registry):
    return PreprocessService(datasets, config.features_config, class_registry)

def test_preprocessing(preprocess_service):
    #Given
    expected_columns = {
        "Time",
        f"rsi_7",
        f"HTF240_rsi_7",
    }
    
    # When
    result = preprocess_service.preprocess_data()
    actual_columns = set(result.columns)
    
    # Then
    assert (
        actual_columns == expected_columns
    ), f"Column mismatch! Expected: {expected_columns}, but got: {actual_columns}"

    # print(feature_df_merged.head())
