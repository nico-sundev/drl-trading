from pandas import DataFrame
import pytest
from ai_trading.data_import.local_data_import_service import LocalDataImportService
from ai_trading.preprocess import multi_tf_preprocessing as pp


@pytest.fixture
def import_svc():
    return LocalDataImportService()
    


def test_import(import_svc: LocalDataImportService):
    print(f"Running test_import with LocalDataImportService")
    imported_df: dict = import_svc.import_data()
    assert len(imported_df) == 3
    assert len(imported_df['H1']) == 100_000
    
