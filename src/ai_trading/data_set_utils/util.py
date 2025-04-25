import logging  # Add logging
from typing import List  # Add Optional

import pandas as pd  # Add pandas
from pandas import DataFrame  # Add DataFrame

from ai_trading.model.asset_price_dataset import AssetPriceDataSet
from ai_trading.model.computed_dataset_container import ComputedDataSetContainer

logger = logging.getLogger(__name__)  # Add logger


def ensure_datetime_time_column(
    df: DataFrame, df_description: str = "DataFrame"
) -> DataFrame:
    """
    Ensures a DataFrame has a 'Time' column of datetime dtype.

    Checks if 'Time' column exists. If not, tries to reset a DatetimeIndex.
    If 'Time' exists but is not datetime, tries to convert it.
    Raises ValueError if 'Time' cannot be created or converted.

    Args:
        df: The input DataFrame.
        df_description: A description of the DataFrame for logging purposes.

    Returns:
        DataFrame: A copy of the input DataFrame with a validated 'Time' column.

    Raises:
        ValueError: If a valid 'Time' column cannot be ensured.
    """
    df_copy = df.copy()
    if "Time" not in df_copy.columns:
        if isinstance(df_copy.index, pd.DatetimeIndex):
            logger.debug(
                f"Resetting DatetimeIndex to create 'Time' column for {df_description}."
            )
            # Reset index but rename the resulting column to 'Time' instead of default 'index'
            df_copy = df_copy.reset_index(names="Time")
        else:
            msg = f"{df_description} must have a 'Time' column or a DatetimeIndex."
            logger.error(msg)
            raise ValueError(msg)

    if not pd.api.types.is_datetime64_any_dtype(df_copy["Time"]):
        logger.debug(
            f"Attempting to convert 'Time' column to datetime for {df_description}."
        )
        try:
            df_copy["Time"] = pd.to_datetime(df_copy["Time"])
        except Exception as e:
            msg = (
                f"Could not convert 'Time' column to datetime for {df_description}: {e}"
            )
            logger.error(msg)
            raise ValueError(msg) from e

    return df_copy


def separate_asset_price_datasets(datasets: List[AssetPriceDataSet]) -> tuple:
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

    return base_dataset, other_datasets


def separate_computed_datasets(datasets: List[ComputedDataSetContainer]) -> tuple:
    """
    Separates the ComputedDataSetContainer datasets into base and other datasets.
    """
    base_dataset = None
    other_datasets = []

    for dataset in datasets:
        if dataset.source_dataset.base_dataset:
            base_dataset = dataset
        else:
            other_datasets.append(dataset)

    return base_dataset, other_datasets


def detect_timeframe(df: DataFrame) -> pd.Timedelta:
    """Auto-detects the timeframe of a dataset."""
    return pd.Timedelta(str(df["Time"].diff().mode()[0]))
