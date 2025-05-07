import os

import pytest

from ai_trading.config.config_loader import ConfigLoader
from ai_trading.config.feature_config_factory import FeatureConfigFactory
from ai_trading.data_import.data_import_manager import DataImportManager
from ai_trading.data_import.local.csv_data_import_service import CsvDataImportService
from ai_trading.data_set_utils.merge_service import MergeService
from ai_trading.model.symbol_import_container import SymbolImportContainer
from ai_trading.preprocess.feature.feature_aggregator import FeatureAggregator
from ai_trading.preprocess.feature.feature_class_registry import FeatureClassRegistry
from ai_trading.preprocess.preprocess_service import PreprocessService


@pytest.fixture
def feature_config_factory():
    """Create a fresh feature config factory instance for testing."""
    factory = FeatureConfigFactory()
    factory.discover_config_classes()
    return factory


@pytest.fixture
def config(feature_config_factory):
    config = ConfigLoader.get_config(
        os.path.join(
            os.path.dirname(__file__), "../resources/applicationConfig-test.json"
        )
    )
    # To speed up the test, only use RSI
    config.features_config.feature_definitions = [
        f for f in config.features_config.feature_definitions if f.name == "rsi"
    ]

    # Parse parameters using the factory
    config.features_config.parse_all_parameters(feature_config_factory)

    return config


@pytest.fixture
def symbol_container(config) -> SymbolImportContainer:
    # Create a service with the complete config
    repository = CsvDataImportService(config.local_data_import_config)
    importer = DataImportManager(repository)

    # Get all symbol containers
    return importer.get_data()[0]


@pytest.fixture
def class_registry():
    return FeatureClassRegistry()


@pytest.fixture
def feast_service(config, symbol_container, mocked_feature_store):
    """Create a real FeastService for testing."""
    from ai_trading.preprocess.feast.feast_service import FeastService

    return FeastService(
        config=config.feature_store_config,
        symbol=symbol_container.symbol,
        asset_data=symbol_container.datasets[0],
    )


@pytest.fixture
def feature_aggregator(config, class_registry, feast_service) -> FeatureAggregator:
    return FeatureAggregator(config, class_registry, feast_service)


@pytest.fixture
def merge_service(config, class_registry, feature_aggregator, feast_service):
    return MergeService()


@pytest.fixture
def preprocess_service(config, class_registry, feature_aggregator, merge_service):
    return PreprocessService(
        features_config=config.features_config,
        feature_class_registry=class_registry,
        feature_aggregator=feature_aggregator,
        merge_service=merge_service,
    )


def test_preprocessing(
    preprocess_service: PreprocessService, symbol_container: SymbolImportContainer
):
    """Test that preprocessing creates the expected feature columns."""
    # Given
    expected_context_related_columns = ["Time", "High", "Low", "Close", "Volume", "Atr"]

    expected_feature_columns = [
        "rsi_7",
        "HTF240_rsi_7",
    ]

    all_expected_columns = sorted(
        expected_context_related_columns + expected_feature_columns
    )

    # When
    result = preprocess_service.preprocess_data(symbol_container)
    actual_columns = sorted(set(result.columns))

    # Then
    assert (
        actual_columns == all_expected_columns
    ), f"Column mismatch! Expected: {all_expected_columns}, but got: {actual_columns}"

    # print(feature_df_merged.head())
