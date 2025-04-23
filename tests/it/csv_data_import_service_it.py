import os

import pytest

from ai_trading.config.local_data_import_config import (
    LocalDataImportConfig,
    SymbolConfig,
)
from ai_trading.data_import.local.csv_data_import_service import CsvDataImportService
from ai_trading.model.asset_price_import_properties import AssetPriceImportProperties


@pytest.fixture
def sample_csv_service():
    """Fixture for testing CSV import."""
    import_properties_objects = [
        AssetPriceImportProperties(
            timeframe="H1",
            base_dataset=True,
            file_path=os.path.join(
                os.path.dirname(__file__), "../resources/test_H1.csv"
            ),
        ),
        AssetPriceImportProperties(
            timeframe="H4",
            base_dataset=False,
            file_path=os.path.join(
                os.path.dirname(__file__), "../resources/test_H4.csv"
            ),
        ),
    ]

    # Create a symbol config with the test data
    symbol_config = SymbolConfig(symbol="TEST", datasets=import_properties_objects)

    # Create a local data import config with the test symbol
    config = LocalDataImportConfig(symbols=[symbol_config])

    return CsvDataImportService(config)


def test_csv_import(sample_csv_service: CsvDataImportService):
    """Test CSV import functionality."""

    symbol_containers = sample_csv_service.import_data(limit=1)

    # Check we have at least one symbol container
    assert len(symbol_containers) > 0

    # Get the first symbol container
    symbol_container = symbol_containers[0]

    # Check the symbol is set correctly
    assert symbol_container.symbol == "TEST"

    # Check that datasets are imported correctly
    assert len(symbol_container.datasets) > 0
    assert any(
        dataset.timeframe == "H1" and not dataset.asset_price_dataset.empty
        for dataset in symbol_container.datasets
    )
