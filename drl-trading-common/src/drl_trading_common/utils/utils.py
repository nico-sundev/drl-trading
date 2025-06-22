import hashlib
import json
import logging

import pandas as pd
from pandas import DataFrame

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

def compute_feature_config_hash(feature_definitions: list[dict]) -> str:
    # Normalize and encode
    json_bytes = json.dumps(feature_definitions, sort_keys=True).encode('utf-8')
    return hashlib.sha256(json_bytes).hexdigest()[:10]  # shorten to 10 chars
