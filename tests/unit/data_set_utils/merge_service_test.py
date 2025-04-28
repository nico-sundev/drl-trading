from typing import List, Tuple

import pandas as pd
import pytest

from ai_trading.data_set_utils.merge_service import MergeService
from ai_trading.data_set_utils.util import separate_computed_datasets
from ai_trading.model.asset_price_dataset import AssetPriceDataSet
from ai_trading.model.computed_dataset_container import ComputedDataSetContainer
from tests.unit.fixture.sample_data import mock_ohlcv_data_1h, mock_ohlcv_data_4h


@pytest.fixture
def sample_data() -> List[AssetPriceDataSet]:
    """Return sample datasets for 1h and 4h timeframes."""
    return [mock_ohlcv_data_1h(True), mock_ohlcv_data_4h(False)]


@pytest.fixture
def prepared_dataframes(
    sample_data: List[AssetPriceDataSet],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare base and higher timeframe dataframes for testing."""
    # Create computed datasets from sample data
    computed_datasets: List[ComputedDataSetContainer] = [
        ComputedDataSetContainer(dataset, dataset.asset_price_dataset)
        for dataset in sample_data
    ]

    # Separate into base and other datasets
    base, other = separate_computed_datasets(computed_datasets)

    # Get the DataFrames
    df_base = base.computed_dataframe
    df_higher = other[0].computed_dataframe

    # Add a custom feature to the higher timeframe for testing
    df_higher["cstm_feature_1"] = df_higher["Close"] + 100
    df_higher["cstm_feature_2"] = df_higher["High"] + 100
    df_higher["cstm_feature_3"] = df_higher["Low"] + 100

    return df_base, df_higher


def test_merge_timeframes(
    prepared_dataframes: Tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Test merging of higher timeframe features into base timeframe.

    Verifies that custom features from higher timeframes are correctly
    mapped to the corresponding rows in the base timeframe.
    """
    # Given
    df_base, df_higher = prepared_dataframes
    merger = MergeService()

    # When
    df_merged = merger.merge_timeframes(df_base, df_higher)

    # Then
    # Verify that the higher timeframe feature is correctly mapped
    assert df_merged.columns.tolist() == [
        "Time",
        "HTF240_cstm_feature_1",
        "HTF240_cstm_feature_2",
        "HTF240_cstm_feature_3",
    ]
    assert df_merged.iloc[0]["HTF240_cstm_feature_1"] == 101.3820
    assert df_merged.iloc[-1]["HTF240_cstm_feature_1"] == 101.4100
