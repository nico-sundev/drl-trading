import logging
from typing import List, Optional

import pandas as pd
from pandas import DataFrame

from ai_trading.common.model.asset_price_dataset import AssetPriceDataSet
from ai_trading.common.model.computed_dataset_container import ComputedDataSetContainer

logger = logging.getLogger(__name__)


def ensure_datetime_index(
    df: DataFrame, df_description: str = "DataFrame"
) -> DataFrame:
    """
    Ensures a DataFrame has a DatetimeIndex.

    If 'Time' column exists, it sets this as the index.
    If no 'Time' column exists but index is already DatetimeIndex, returns as is.
    Raises ValueError if a proper datetime index cannot be created.

    Args:
        df: The input DataFrame.
        df_description: A description of the DataFrame for logging purposes.

    Returns:
        DataFrame: A copy of the input DataFrame with a DatetimeIndex.

    Raises:
        ValueError: If a valid DatetimeIndex cannot be ensured.
    """
    df_copy = df.copy()

    # Case 1: DataFrame already has a DatetimeIndex
    if isinstance(df_copy.index, pd.DatetimeIndex):
        logger.debug(f"{df_description} already has DatetimeIndex.")
        return df_copy

    # Case 2: DataFrame has a 'Time' column that can be made into an index
    if "Time" in df_copy.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_copy["Time"]):
            logger.debug(f"Converting 'Time' column to datetime for {df_description}.")
            try:
                df_copy["Time"] = pd.to_datetime(df_copy["Time"])
            except Exception as e:
                msg = f"Could not convert 'Time' column to datetime for {df_description}: {e}"
                logger.error(msg)
                raise ValueError(msg) from e

        logger.debug(f"Setting 'Time' column as DatetimeIndex for {df_description}.")
        return df_copy.set_index("Time")

    # Case 3: No suitable time column or index found
    msg = f"{df_description} must have a 'Time' column or a DatetimeIndex."
    logger.error(msg)
    raise ValueError(msg)


def ensure_datetime_time_column(
    df: DataFrame, df_description: str = "DataFrame"
) -> DataFrame:
    """
    Ensures a DataFrame has a 'Time' column of datetime dtype.

    This is a backward compatibility function. New code should use ensure_datetime_index instead.

    If the DataFrame has a DatetimeIndex, it resets the index to make a 'Time' column.
    If 'Time' exists but is not datetime, tries to convert it.

    Args:
        df: The input DataFrame.
        df_description: A description of the DataFrame for logging purposes.

    Returns:
        DataFrame: A copy of the input DataFrame with a validated 'Time' column.

    Raises:
        ValueError: If a valid 'Time' column cannot be ensured.
    """
    df_copy = df.copy()

    # Case 1: DataFrame has a DatetimeIndex that needs to be reset to a column
    if isinstance(df_copy.index, pd.DatetimeIndex):
        logger.debug(
            f"Resetting DatetimeIndex to create 'Time' column for {df_description}."
        )
        return df_copy.reset_index(names="Time")

    # Case 2: DataFrame already has a Time column that might need type conversion
    if "Time" in df_copy.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_copy["Time"]):
            logger.debug(f"Converting 'Time' column to datetime for {df_description}.")
            try:
                df_copy["Time"] = pd.to_datetime(df_copy["Time"])
            except Exception as e:
                msg = f"Could not convert 'Time' column to datetime for {df_description}: {e}"
                logger.error(msg)
                raise ValueError(msg) from e
        return df_copy

    # Case 3: No suitable time column or index found
    msg = f"{df_description} must have a 'Time' column or a DatetimeIndex."
    logger.error(msg)
    raise ValueError(msg)


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
        time_diffs = df["Time"].diff().dropna()
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
