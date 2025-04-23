import os

import pytest

from ai_trading.config.config_loader import ConfigLoader
from ai_trading.config.feature_config_registry import FeatureConfigRegistry
from ai_trading.data_import.data_import_manager import DataImportManager
from ai_trading.data_import.local.csv_data_import_service import CsvDataImportService
from ai_trading.preprocess.feature.feature_aggregator import FeatureAggregator
from ai_trading.preprocess.feature.feature_class_registry import FeatureClassRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    FeatureConfigRegistry._instance = None


@pytest.fixture(autouse=True)
def config(reset_registry):
    return ConfigLoader.get_config(
        os.path.join(
            os.path.dirname(__file__), "../resources/applicationConfig-test.json"
        )
    )


@pytest.fixture
def dataset(config):
    all_datasets = []

    # Create a service with the complete config
    repository = CsvDataImportService(config.local_data_import_config)
    importer = DataImportManager(repository)

    # Get all symbol containers
    symbol_containers = importer.get_data(100)

    # Extract datasets from all symbols
    for symbol_container in symbol_containers:
        all_datasets.extend(symbol_container.datasets)

    return [dataset for dataset in all_datasets if dataset.timeframe == "H1"][0]


@pytest.fixture(autouse=True)
def class_registry():
    return FeatureClassRegistry()


@pytest.fixture
def feature_aggregator(dataset, config, class_registry):
    return FeatureAggregator(dataset, config.features_config, class_registry)


def test_features(feature_aggregator: FeatureAggregator):
    # Given
    expected_columns = ["Time", "rsi_7"]

    # When
    result_df = feature_aggregator.compute()

    # Then
    assert set(expected_columns).issubset(set(result_df.columns))
