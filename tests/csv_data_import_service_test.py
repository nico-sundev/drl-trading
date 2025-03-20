import os
import pytest
import pandas as pd

from ai_trading.data_import.local.csv_data_import_service import CsvDataImportService


@pytest.fixture
def sample_csv_service():
    """Fixture for testing CSV import."""
    file_paths = {
        "H1": os.path.join(os.path.dirname(__file__), "resources/test_H1.csv"),
        "H4": os.path.join(os.path.dirname(__file__), "resources/test_H4.csv"),
    }
    return CsvDataImportService(file_paths)


def test_csv_import(sample_csv_service):
    """Test CSV import functionality."""

    data = sample_csv_service.import_data(limit=1)
    assert "H1" in data
    assert not data["H1"].empty
