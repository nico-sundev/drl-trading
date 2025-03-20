import os
import pandas as pd
import pytest
from ai_trading.data_import.local.csv_data_import_service import CsvDataImportService
from ai_trading.preprocess.merging_service import MergingService

@pytest.fixture
def sample_data():
    """Fixture for testing CSV import."""
    file_paths = {
        "H1": os.path.join(os.path.dirname(__file__), "../resources/test_H1.csv"),
        "H4": os.path.join(os.path.dirname(__file__), "../resources/test_H4.csv"),
    }
    svc = CsvDataImportService(file_paths)
    data = svc.import_data()
    return [data["H1"], data["H4"]]

def test_merge_timeframes(sample_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_30m, df_4h = sample_data
    merger: MergingService = MergingService(df_30m, df_4h)
    df_merged: pd.DataFrame = merger.merge_timeframes()

    assert df_merged.iloc[0]["HTF_Close"] == 1.38485
    assert df_merged.iloc[-1]["HTF_Close"] == 1.38155


def test_timeframe_detection() -> None:
    df: pd.DataFrame = pd.DataFrame({"Time": pd.date_range("2024-03-19 00:00:00", periods=4, freq="1H")})
    merger: MergingService = MergingService(df, df)
    detected_tf: pd.Timedelta = merger.detect_timeframe(df)
    
    assert detected_tf == pd.Timedelta("1H")