import os
import pandas as pd
import pytest

from ai_trading.config.config_loader import ConfigLoader
from ai_trading.config.feature_config_registry import FeatureConfigRegistry
from ai_trading.data_import.data_import_manager import DataImportManager
from ai_trading.data_import.local.csv_data_import_service import CsvDataImportService
from ai_trading.preprocess.feature.feature_aggregator import FeatureAggregator
from ai_trading.preprocess.feature.feature_class_registry import FeatureClassRegistry
from ai_trading.preprocess.feature.feature_factory import FeatureFactory
from ai_trading.preprocess.merging_service import MergingService


@pytest.fixture
def datasets():
    file_paths = {
        "H1": os.path.join(
            os.path.dirname(__file__), "..\\..\\data\\raw\\EURUSD_H1.csv"
        ),
        "H4": os.path.join(
            os.path.dirname(__file__), "..\\..\\data\\raw\\EURUSD_H4.csv"
        ),
    }
    repository = CsvDataImportService(file_paths)
    importer = DataImportManager(repository)
    return importer.get_data(100)


@pytest.fixture(autouse=True)
def reset_registry():
    FeatureConfigRegistry._instance = None

@pytest.fixture(autouse=True)
def config(reset_registry):
    config = ConfigLoader.get_config(
        os.path.join(
            os.path.dirname(__file__), "../resources/applicationConfig-test.json"
        )
    )
    # To speed up the test, only use RSI
    config.features_config.feature_definitions = [f for f in config.features_config.feature_definitions if f.name == "rsi"]
    return config

@pytest.fixture(autouse=True)
def class_registry():
    return FeatureClassRegistry()

def test_preprocessing(datasets, config, class_registry):
    aggregator_h1 = FeatureAggregator(datasets["H1"], config.features_config, class_registry)
    aggregator_h4 = FeatureAggregator(datasets["H4"], config.features_config, class_registry)
    feature_df_1h = aggregator_h1.compute()
    feature_df_4h = aggregator_h4.compute()

    # Merge Timeframes
    merger: MergingService = MergingService(feature_df_1h, feature_df_4h)
    feature_df_merged: pd.DataFrame = merger.merge_timeframes()

    expected_columns = {
        "Time",
        f"rsi_7",
        f"HTF240_rsi_7",
    }
    actual_columns = set(feature_df_merged.columns)

    assert (
        actual_columns == expected_columns
    ), f"Column mismatch! Expected: {expected_columns}, but got: {actual_columns}"

    # print(feature_df_merged.head())
