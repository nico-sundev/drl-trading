import logging
import time
from pandas import DataFrame
import pytest
from ai_trading.preprocess import multi_tf_preprocessing as pp
from ai_trading.data_import.local_data_import_service import LocalDataImportService


@pytest.fixture
def import_svc():
    return LocalDataImportService()


def test_merge_timeframes(import_svc: LocalDataImportService):
    print(f"Running merge_timeframes")
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    start_time = time.time()
    len = 30_000
    tf_data_sets = import_svc.import_data(3000)
    end_time = time.time()
    execution_time = end_time - start_time
    logging.debug(f"Import data Execution time: {execution_time} seconds")
    assert len(merged_dataframe) == len

    start_time = time.time()
    merged_dataframe: DataFrame = pp.merge_timeframes_into_base(tf_data_sets, True)
    end_time = time.time()
    execution_time = end_time - start_time
    logging.debug(f"Merge timeframes Execution time: {execution_time} seconds")

    logging.debug(merged_dataframe.head(500))
