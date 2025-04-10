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
from ai_trading.preprocess.merging_service import MergingService
from dask import delayed, compute


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
    config.features_config.feature_definitions = [
        f for f in config.features_config.feature_definitions if f.name == "rsi"
    ]
    return config


@pytest.fixture(autouse=True)
def class_registry():
    return FeatureClassRegistry()


def test_preprocessing(datasets, config, class_registry):

    # Parallelize feature computation for each dataset using delayed
    feature_results = [
        delayed(compute_feature)(dataset, config.features_config, class_registry)
        for dataset in datasets
    ]

    # Compute all delayed tasks in parallel
    asset_price_datasets: List[ComputedDataSetContainer] = dask.compute(*feature_results)
    base_dataset = [
        dataset for dataset in asset_price_datasets if dataset.source_dataset.base_dataset
    ][0]
    other_datasets = [
        dataset for dataset in asset_price_datasets if not dataset.source_dataset.base_dataset
    ]

    # Merge Timeframes

    # Initialize a merged dataset with the base_dataset first
    merged_result = base_dataset.computed_dataframe

    # Iterate through other_datasets and merge with base_dataset one by one
    for dataset in other_datasets:
        # Create an instance of the MergingService for each pair of datasets
        merger = MergingService(merged_result, dataset.computed_dataframe)
        # Perform the merge and update the merged_result
        merged_result = merger.merge_timeframes()

    expected_columns = {
        "Time",
        f"rsi_7",
        f"HTF240_rsi_7",
    }
    actual_columns = set(merged_result.columns)

    assert (
        actual_columns == expected_columns
    ), f"Column mismatch! Expected: {expected_columns}, but got: {actual_columns}"

    # print(feature_df_merged.head())


def compute_feature(dataset, config, class_registry):
    """
    Computes the features for a given dataset.
    This is separated out for better clarity and parallelization.
    """
    feature_aggregator = FeatureAggregator(
        dataset.asset_price_dataset, config, class_registry
    )
    computed_data = feature_aggregator.compute()  # Perform the actual computation
    return ComputedDataSetContainer(dataset, computed_data)
