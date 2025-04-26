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
def symbol_container(config):
    # Create a service with the complete config
    repository = CsvDataImportService(config.local_data_import_config)
    importer = DataImportManager(repository)

    # Get all symbol containers
    return importer.get_data(100)[0]


@pytest.fixture
def class_registry():
    return FeatureClassRegistry()


@pytest.fixture
def feast_service(config, symbol_container, init_feast_for_tests):
    """Create a real FeastService for testing."""
    from ai_trading.preprocess.feast.feast_service import FeastService

    return FeastService(
        feature_store_config=config.feature_store_config,
        symbol=symbol_container.symbol,
        asset_data=symbol_container.datasets[0],
    )


@pytest.fixture
def preprocess_service(config, class_registry, feast_service):
    return PreprocessService(
        features_config=config.features_config,
        feature_class_registry=class_registry,
        feast_service=feast_service,
    )


def test_preprocessing(preprocess_service, symbol_container):
    """Test that preprocessing creates the expected feature columns."""
    # Given
    expected_columns = {
        "Time",
        "rsi_7",
        "HTF240_rsi_7",
    }

    # When
    result = preprocess_service.preprocess_data(symbol_container)
    actual_columns = set(result.columns)

    # Then
    assert (
        actual_columns == expected_columns
    ), f"Column mismatch! Expected: {expected_columns}, but got: {actual_columns}"

    # print(feature_df_merged.head())
