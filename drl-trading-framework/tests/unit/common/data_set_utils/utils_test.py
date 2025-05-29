from unittest.mock import MagicMock

import pandas as pd
import pytest

from drl_trading_framework.common.model.asset_price_dataset import AssetPriceDataSet
from drl_trading_framework.common.model.computed_dataset_container import (
    ComputedDataSetContainer,
)
from drl_trading_framework.preprocess.data_set_utils.util import (
    detect_timeframe,
    separate_computed_datasets,
)


def test_separate_base_and_other_datasets():
    # Mock datasets
    base_dataset = MagicMock(spec=AssetPriceDataSet)
    base_dataset.base_dataset = True

    other_dataset_1 = MagicMock(spec=AssetPriceDataSet)
    other_dataset_1.base_dataset = False

    other_dataset_2 = MagicMock(spec=AssetPriceDataSet)
    other_dataset_2.base_dataset = False

    base_dataset_container = ComputedDataSetContainer(
        source_dataset=base_dataset, computed_dataframe=MagicMock()
    )
    other_dataset_container_1 = ComputedDataSetContainer(
        source_dataset=other_dataset_1, computed_dataframe=MagicMock()
    )
    other_dataset_container_2 = ComputedDataSetContainer(
        source_dataset=other_dataset_2, computed_dataframe=MagicMock()
    )

    datasets = [
        base_dataset_container,
        other_dataset_container_1,
        other_dataset_container_2,
    ]

    # Call the function
    result_base, result_others = separate_computed_datasets(datasets)

    # Assertions
    assert result_base == base_dataset_container
    assert other_dataset_container_1 in result_others
    assert other_dataset_container_2 in result_others
    assert len(result_others) == 2


def test_timeframe_detection() -> None:
    # Given
    df: pd.DataFrame = pd.DataFrame(
        {"Time": pd.date_range("2024-03-19 00:00:00", periods=4, freq="1h")}
    )

    # When
    detected_tf: pd.Timedelta = detect_timeframe(df)

    # Then
    assert detected_tf == pd.Timedelta("1h")


def test_timeframe_detection_empty_dataframe():
    """Test detect_timeframe with an empty DataFrame."""
    # Given
    empty_df = pd.DataFrame(columns=["Time"])

    # When/Then
    with pytest.raises(ValueError, match="DataFrame has less than 2 rows"):
        detect_timeframe(empty_df)


def test_timeframe_detection_none_dataframe():
    """Test detect_timeframe with None instead of a DataFrame."""
    # Given
    none_df = None

    # When/Then
    with pytest.raises(ValueError, match="DataFrame is None"):
        detect_timeframe(none_df)


def test_timeframe_detection_no_time_column_and_index():
    """Test detect_timeframe with a DataFrame missing the Time column."""
    # Given
    df_no_time = pd.DataFrame({"Value": [1, 2, 3]})

    # When/Then
    with pytest.raises(ValueError):
        detect_timeframe(df_no_time)


def test_timeframe_detection_single_row():
    """Test detect_timeframe with a DataFrame containing only one row."""
    # Given
    single_row_df = pd.DataFrame({"Time": [pd.Timestamp("2023-01-01")]})

    # When/Then
    with pytest.raises(ValueError, match="less than 2 rows"):
        detect_timeframe(single_row_df)
