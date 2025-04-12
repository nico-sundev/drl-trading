import os
import pytest
from ai_trading.data_import.local.csv_data_import_service import CsvDataImportService
from ai_trading.model.asset_price_import_properties import AssetPriceImportProperties


@pytest.fixture
def sample_csv_service():
    """Fixture for testing CSV import."""
    file_paths = [
        {
            "timeframe": "H1",
            "base_dataset": True,
            "file_path": os.path.join(
                os.path.dirname(__file__), "../resources/test_H1.csv"
            ),
        },
        {
            "timeframe": "H4",
            "base_dataset": False,
            "file_path": os.path.join(
                os.path.dirname(__file__), "../resources/test_H4.csv"
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

    return CsvDataImportService(import_properties_objects)


def test_csv_import(sample_csv_service: CsvDataImportService):
    """Test CSV import functionality."""

    data = sample_csv_service.import_data(limit=1)
    timeframes = [dataset.timeframe for dataset in data]
    assert any(dataset.timeframe == "H1" and not dataset.asset_price_dataset.empty for dataset in data)
