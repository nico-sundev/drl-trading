from pandas import DataFrame
import pytest
from ai_trading.preprocess import multi_tf_preprocessing as pp
from ai_trading.data_import.local_data_import_service import LocalDataImportService


@pytest.fixture
def import_svc():
    return LocalDataImportService()


def test_merge_timeframes(import_svc: LocalDataImportService):
    print(f"Running merge_timeframes")
    merged_dataframe: DataFrame = pp.merge_timeframes_into_base(import_svc.import_data())
    assert len(merged_dataframe) == 100_000
    print(merged_dataframe.head(100))
    assert len(merged_dataframe.columns) == 14
    
