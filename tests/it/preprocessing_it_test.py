import os
from typing import List
import dask
import pandas as pd
import pytest

from ai_trading.config.config_loader import ConfigLoader
from ai_trading.config.feature_config_registry import FeatureConfigRegistry
from ai_trading.data_import.data_import_manager import DataImportManager
from ai_trading.data_import.local.csv_data_import_service import CsvDataImportService
from ai_trading.model.asset_price_dataset import AssetPriceDataSet
from ai_trading.model.asset_price_import_properties import AssetPriceImportProperties
from ai_trading.model.computed_dataset_container import ComputedDataSetContainer
from ai_trading.preprocess.feature.feature_aggregator import FeatureAggregator
from ai_trading.preprocess.feature.feature_class_registry import FeatureClassRegistry
from ai_trading.preprocess.feature.feature_factory import FeatureFactory
from ai_trading.data_set_utils.merge_service import MergeService
from dask import delayed, compute

from ai_trading.preprocess.preprocess_service import PreprocessService


@pytest.fixture
def datasets():
    file_paths = [
        {
            "timeframe": "H1",
            "base_dataset": True,
            "file_path": os.path.join(
                os.path.dirname(__file__), "..\\..\\data\\raw\\EURUSD_H1.csv"
            ),
        },
        {
            "timeframe": "H4",
            "base_dataset": False,
            "file_path": os.path.join(
                os.path.dirname(__file__), "..\\..\\data\\raw\\EURUSD_H4.csv"
            ),
        },
    ]
    import_properties_objects = [
        AssetPriceImportProperties(
            timeframe=item["timeframe"],
            base_dataset=item["base_dataset"],
            file_path=item["file_path"],
        )
        for item in file_paths
    ]

    repository = CsvDataImportService(import_properties_objects)
    importer = DataImportManager(repository)
    return importer.get_data(100)


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
    return config.features_config


@pytest.fixture
def class_registry():
    return FeatureClassRegistry()


@pytest.fixture
def preprocess_service(datasets, config, class_registry):
    return PreprocessService(datasets, config, class_registry)

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
