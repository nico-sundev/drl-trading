from datetime import timedelta

import pandas as pd
import pytest
from drl_trading_common.models.timeframe import Timeframe

from drl_trading_core.common.model.asset_price_dataset import AssetPriceDataSet
from drl_trading_core.preprocess.data_set_utils.strip_service import StripService


@pytest.fixture
def base_dataset(mock_ohlcv_data_1h):
    """Base dataset fixture with 1-hour timeframe."""
    df = mock_ohlcv_data_1h.asset_price_dataset
    return df


@pytest.fixture
def higher_timeframe_dataset(mock_ohlcv_data_4h):
    """Higher timeframe dataset fixture with 4-hour timeframe."""
    df = mock_ohlcv_data_4h.asset_price_dataset
    return df


@pytest.fixture
def extended_higher_timeframe_dataset(higher_timeframe_dataset):
    """Higher timeframe dataset with extended range before and after base dataset."""
    # Original dataset with index reset to get Time as a column
    df = higher_timeframe_dataset.reset_index(names=["Time"])

    # Add rows with earlier timestamps (before base dataset starts)
    first_time = df["Time"].min()
    early_times = [
        first_time - timedelta(hours=12),
        first_time - timedelta(hours=8),
        first_time - timedelta(hours=4),
    ]
    early_data = []
    for t in early_times:
        row = df.iloc[0].copy()
        row["Time"] = t
        early_data.append(row)
    early_df = pd.DataFrame(
        early_data
    )  # Add rows with later timestamps (after base dataset ends)
    last_time = df["Time"].max()
    late_times = [
        last_time + timedelta(hours=4),
        last_time + timedelta(hours=8),
        last_time + timedelta(hours=12),
    ]
    late_data = []
    for t in late_times:
        row = df.iloc[-1].copy()
        row["Time"] = t
        late_data.append(row)
    late_df = pd.DataFrame(late_data)  # Combine all data
    extended_df = pd.concat([early_df, df, late_df]).sort_values("Time")

    # Set the datetime index
    extended_df = extended_df.set_index("Time")
    return extended_df


def test_strip_higher_timeframes_end(base_dataset, higher_timeframe_dataset):
    """Test that strip_higher_timeframes correctly strips data at the end."""
    # Given
    service = StripService()
    base_start_timestamp = base_dataset.index[0]
    base_end_timestamp = base_dataset.index[-1]

    # When
    stripped_df = service.strip_higher_timeframes(
        base_start_timestamp, base_end_timestamp, higher_timeframe_dataset
    )

    # Then
    assert not stripped_df.empty, "The stripped dataframe should not be empty."
    assert stripped_df.index[-1] < base_end_timestamp + timedelta(
        hours=8
    ), "The last timestamp in the stripped dataframe should be less than the base end timestamp + 2*timeframe_duration."


def test_strip_higher_timeframes_beginning_and_end(
    base_dataset, extended_higher_timeframe_dataset
):
    """Test that strip_higher_timeframes correctly strips data at both beginning and end."""
    # Given
    service = StripService()
    base_start_timestamp = base_dataset.index[0]
    base_end_timestamp = base_dataset.index[-1]

    # When
    stripped_df = service.strip_higher_timeframes(
        base_start_timestamp, base_end_timestamp, extended_higher_timeframe_dataset
    )

    # Then
    assert not stripped_df.empty, "The stripped dataframe should not be empty."

    # Check beginning stripping
    assert stripped_df.index[0] >= base_start_timestamp - timedelta(
        hours=8
    ), "The first timestamp should not be earlier than base start timestamp - 2*timeframe_duration."

    # Check end stripping
    assert stripped_df.index[-1] < base_end_timestamp + timedelta(
        hours=8
    ), "The last timestamp should be less than base end timestamp + 2*timeframe_duration."

    # Verify we actually removed some rows
    assert len(stripped_df) < len(
        extended_higher_timeframe_dataset
    ), "The stripped dataframe should have fewer rows than the original extended dataframe."


def test_strip_asset_price_datasets_complete(
    base_dataset, extended_higher_timeframe_dataset
):
    """Test strip_asset_price_datasets with both beginning and end stripping."""
    # Given
    service = StripService()
    datasets = [
        AssetPriceDataSet("H1", True, base_dataset),
        AssetPriceDataSet("H4", False, extended_higher_timeframe_dataset),
    ]

    base_start_timestamp = base_dataset.index[0]
    base_end_timestamp = base_dataset.index[-1]

    # When
    stripped_datasets = service.strip_asset_price_datasets(datasets)

    # Then
    assert len(stripped_datasets) == 2, "The number of datasets should remain the same."

    stripped_base_dataset = stripped_datasets[0]
    stripped_higher_timeframe_dataset = stripped_datasets[1]

    assert (
        stripped_base_dataset.timeframe == Timeframe.HOUR_1
    ), "The base dataset should retain its timeframe."
    assert (
        stripped_higher_timeframe_dataset.timeframe == Timeframe.HOUR_4
    ), "The higher timeframe dataset should retain its timeframe."

    # Ensure the stripped dataset has a DatetimeIndex
    stripped_df = stripped_higher_timeframe_dataset.asset_price_dataset
    assert isinstance(
        stripped_df.index, pd.DatetimeIndex
    ), "The result should have a DatetimeIndex"

    # Check beginning stripping
    assert stripped_df.index[0] >= base_start_timestamp - timedelta(
        hours=8
    ), "The first timestamp should not be earlier than base start timestamp - 2*timeframe_duration."

    # Check end stripping
    assert stripped_df.index[-1] < base_end_timestamp + timedelta(
        hours=8
    ), "The last timestamp should be less than base end timestamp + 2*timeframe_duration."
