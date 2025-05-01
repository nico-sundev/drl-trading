import os

import pytest

from ai_trading.config.config_loader import ConfigLoader
from ai_trading.config.feature_config_registry import FeatureConfigRegistry
from ai_trading.data_import.data_import_manager import DataImportManager
from ai_trading.data_import.local.csv_data_import_service import CsvDataImportService
from ai_trading.preprocess.feast.feast_service import FeastService
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

    # Filter to get H1 timeframe datasets
    h1_datasets = [dataset for dataset in all_datasets if dataset.timeframe == "H1"]
    return h1_datasets[0]


@pytest.fixture(autouse=True)
def class_registry():
    return FeatureClassRegistry()


@pytest.fixture
def feast_service(config):
    """Create a FeastService instance for testing."""
    return FeastService(
        config.feature_store_config
    )  # Access feature_store_config directly from config


@pytest.fixture
def feature_aggregator(config, class_registry, feast_service):
    """Create a FeatureAggregator instance for testing."""
    return FeatureAggregator(
        config=config.features_config,
        class_registry=class_registry,
        feast_service=feast_service,
    )


def test_features(feature_aggregator: FeatureAggregator, dataset):
    # Given
    expected_columns = ["Time", "rsi_7"]
    symbol = "EURUSD"  # Assuming this is the symbol for the test dataset

    # When
    # Get delayed tasks from compute
    delayed_tasks = feature_aggregator.compute(asset_data=dataset, symbol=symbol)

    # Execute the delayed tasks using dask.compute
    import dask

    computed_results = dask.compute(*delayed_tasks)

    # Filter out None results
    computed_dfs = [df for df in computed_results if df is not None]

    # Combine results (simplified for test - just checking the first computed dataframe)
    result_df = computed_dfs[0] if computed_dfs else None

    # Then
    assert result_df is not None
    assert set(expected_columns).issubset(set(result_df.columns))
