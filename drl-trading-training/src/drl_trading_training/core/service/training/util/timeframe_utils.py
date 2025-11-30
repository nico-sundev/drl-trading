import logging
from typing import Optional
import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)

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
