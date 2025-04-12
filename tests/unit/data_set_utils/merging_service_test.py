import os
from typing import List
import pandas as pd
import pytest
from ai_trading.data_import.local.csv_data_import_service import CsvDataImportService
from ai_trading.data_set_utils.util import separate_base_and_other_datasets
from ai_trading.model.asset_price_dataset import AssetPriceDataSet
from ai_trading.data_set_utils.merge_service import MergeService
from ai_trading.model.asset_price_import_properties import AssetPriceImportProperties
from ai_trading.model.computed_dataset_container import ComputedDataSetContainer


@pytest.fixture
def sample_data():
    """Fixture for testing CSV import."""
    file_paths = [
        {
            "timeframe": "H1",
            "base_dataset": True,
            "file_path": os.path.join(
                os.path.dirname(__file__), "../../resources/test_H1.csv"
            ),
        },
        {
            "timeframe": "H4",
            "base_dataset": False,
            "file_path": os.path.join(
                os.path.dirname(__file__), "../../resources/test_H4.csv"
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

    repository = CsvDataImportService(import_properties_objects)
    return repository.import_data()


def test_merge_timeframes(sample_data: List[AssetPriceDataSet]) -> None:
    computed_datasets: List[ComputedDataSetContainer] = [
        ComputedDataSetContainer(dataset, dataset.asset_price_dataset)
        for dataset in sample_data
    ]
    base, other = separate_base_and_other_datasets(computed_datasets)
    df_30m = base.computed_dataframe
    df_4h = other[0].computed_dataframe
    # Mock computation of a custom feature for higher TF
    df_4h["cstm_feature_1"] = df_4h["Close"] + 100
    merger: MergeService = MergeService(df_30m, df_4h)
    df_merged: pd.DataFrame = merger.merge_timeframes()

    assert df_merged.iloc[0]["HTF240_cstm_feature_1"] == 101.38485
    assert df_merged.iloc[-1]["HTF240_cstm_feature_1"] == 101.38155


def test_timeframe_detection() -> None:
    df: pd.DataFrame = pd.DataFrame(
        {"Time": pd.date_range("2024-03-19 00:00:00", periods=4, freq="1h")}
    )
    merger: MergeService = MergeService(df, df)
    detected_tf: pd.Timedelta = merger.detect_timeframe(df)

    assert detected_tf == pd.Timedelta("1h")
