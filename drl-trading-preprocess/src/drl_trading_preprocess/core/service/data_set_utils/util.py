import logging
from typing import List, Optional

import pandas as pd
from pandas import DataFrame

from drl_trading_core.common.model.asset_price_dataset import AssetPriceDataSet
from drl_trading_core.common.model.computed_dataset_container import (
    ComputedDataSetContainer,
)

logger = logging.getLogger(__name__)

def separate_asset_price_datasets(
    datasets: List[AssetPriceDataSet],
) -> tuple[AssetPriceDataSet, List[AssetPriceDataSet]]:
    """
    Separates the AssetPriceDataSet datasets into base and other datasets.
    """
    base_dataset = None
    other_datasets = []

    for dataset in datasets:
        if dataset.base_dataset:
            base_dataset = dataset
        else:
            other_datasets.append(dataset)

    if base_dataset is None:
        raise ValueError("No base dataset found in the provided datasets.")
    return base_dataset, other_datasets


def separate_computed_datasets(
    datasets: List[ComputedDataSetContainer],
) -> tuple[ComputedDataSetContainer, List[ComputedDataSetContainer]]:
    """
    Separates the ComputedDataSetContainer datasets into base and other datasets.
    """
    base_dataset = None
    other_datasets: List[ComputedDataSetContainer] = []

    for dataset in datasets:
        if dataset.source_dataset.base_dataset:
            base_dataset = dataset
        else:
            other_datasets.append(dataset)

    if base_dataset is None:
        raise ValueError("No base dataset found in the provided datasets.")

    return base_dataset, other_datasets


def detect_timeframe(df: Optional[DataFrame]) -> pd.Timedelta:
    """
    Auto-detects the timeframe of a dataset based on the most common time difference.

    Works with either a 'Time' column or a DatetimeIndex.

    Args:
        df: DataFrame containing either a DatetimeIndex or a 'Time' column with datetime values

    Returns:
        Timedelta representing the detected timeframe

    Raises:
        ValueError: If df is None, empty, doesn't have datetime values,
                   or doesn't have enough rows to detect timeframe
    """
    if df is None:
        msg = "Cannot detect timeframe: DataFrame is None"
        logger.error(msg)
        raise ValueError(msg)

    if len(df) < 2:
        msg = "Cannot detect timeframe: DataFrame has less than 2 rows"
        logger.error(msg)
        raise ValueError(msg)

    # Handle DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        time_diffs = df.index.to_series().diff().dropna()
    # Handle 'Time' column
    elif "Time" in df.columns:
        time_diffs = pd.to_datetime(df["Time"]).diff().dropna()
    else:
        msg = "Cannot detect timeframe: No DatetimeIndex or 'Time' column found in DataFrame"
        logger.error(msg)
        raise ValueError(msg)

    if len(time_diffs) == 0:
        msg = "Cannot detect timeframe: No valid time differences found"
        logger.error(msg)
        raise ValueError(msg)

    # Get the most common time difference
    most_common_diff = time_diffs.mode()[0]

    # If we get NaT (not a time) result, raise error
    if pd.isna(most_common_diff):
        msg = "Cannot detect timeframe: Invalid time differences detected"
        logger.error(msg)
        raise ValueError(msg)

    return pd.Timedelta(str(most_common_diff))
