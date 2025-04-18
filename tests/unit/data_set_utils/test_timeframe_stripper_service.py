from datetime import timedelta

import pytest

from ai_trading.data_set_utils.timeframe_stripper_service import (
    TimeframeStripperService,
)
from ai_trading.model.asset_price_dataset import AssetPriceDataSet
from tests.unit.fixture.sample_data import mock_ohlcv_data_1h, mock_ohlcv_data_4h


@pytest.fixture
def base_dataset():
    return mock_ohlcv_data_1h().asset_price_dataset


@pytest.fixture
def higher_timeframe_dataset():
    return mock_ohlcv_data_4h().asset_price_dataset


def test_strip_higher_timeframes(base_dataset, higher_timeframe_dataset):
    # Arrange
    service = TimeframeStripperService()
    base_end_timestamp = base_dataset["Time"].iloc[
        -1
    ]  # Last timestamp of the base dataset

    # Act
    stripped_df = service.strip_higher_timeframes(
        base_end_timestamp, higher_timeframe_dataset
    )

    # Assert
    assert not stripped_df.empty, "The stripped dataframe should not be empty."
    assert stripped_df["Time"].iloc[-1] < base_end_timestamp + timedelta(
        hours=8
    ), "The last timestamp in the stripped dataframe should be less than the base end timestamp + timeframe duration."


def test_strip_asset_price_datasets(base_dataset, higher_timeframe_dataset):
    # Arrange
    service = TimeframeStripperService()
    datasets = [
        AssetPriceDataSet("H1", True, base_dataset),
        AssetPriceDataSet("H4", False, higher_timeframe_dataset),
    ]

    # Act
    stripped_datasets = service.strip_asset_price_datasets(datasets)

    # Assert
    assert len(stripped_datasets) == 2, "The number of datasets should remain the same."

    stripped_base_dataset = stripped_datasets[0]
    stripped_higher_timeframe_dataset = stripped_datasets[1]

    assert (
        stripped_base_dataset.timeframe == "H1"
    ), "The base dataset should retain its timeframe."
    assert (
        stripped_higher_timeframe_dataset.timeframe == "H4"
    ), "The higher timeframe dataset should retain its timeframe."

    # Ensure the higher timeframe dataset is stripped correctly
    base_end_timestamp = base_dataset["Time"].iloc[-1]
    assert stripped_higher_timeframe_dataset.asset_price_dataset["Time"].iloc[
        -1
    ] < base_end_timestamp + timedelta(
        hours=8
    ), "The last timestamp in the stripped higher timeframe dataset should be less than the base end timestamp + timeframe duration."
