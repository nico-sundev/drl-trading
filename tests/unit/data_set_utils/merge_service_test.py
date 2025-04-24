from typing import List

import pandas as pd
import pytest

from ai_trading.data_set_utils.merge_service import MergeService
from ai_trading.data_set_utils.util import separate_computed_datasets
from ai_trading.model.asset_price_dataset import AssetPriceDataSet
from ai_trading.model.computed_dataset_container import ComputedDataSetContainer
from tests.unit.fixture.sample_data import mock_ohlcv_data_1h, mock_ohlcv_data_4h


@pytest.fixture
def sample_data():
    return [mock_ohlcv_data_1h(True), mock_ohlcv_data_4h(False)]


def test_merge_timeframes(sample_data: List[AssetPriceDataSet]) -> None:
    computed_datasets: List[ComputedDataSetContainer] = [
        ComputedDataSetContainer(dataset, dataset.asset_price_dataset)
        for dataset in sample_data
    ]
    base, other = separate_computed_datasets(computed_datasets)
    df_30m = base.computed_dataframe
    df_4h = other[0].computed_dataframe
    # Mock computation of a custom feature for higher TF
    df_4h["cstm_feature_1"] = df_4h["Close"] + 100
    merger: MergeService = MergeService(df_30m, df_4h)
    df_merged: pd.DataFrame = merger.merge_timeframes()

    assert df_merged.iloc[0]["HTF240_cstm_feature_1"] == 101.3820
    assert df_merged.iloc[-1]["HTF240_cstm_feature_1"] == 101.4100
