import abc
import logging
from typing import Dict, Optional

import pandas as pd

from ai_trading.preprocess.data_set_utils.util import (
    detect_timeframe,
    ensure_datetime_index,
)


class MergeServiceInterface(abc.ABC):
    """Interface for services that merge higher timeframe data into lower timeframe datasets."""

    @abc.abstractmethod
    def merge_timeframes(
        self, base_df: pd.DataFrame, higher_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges features from a higher timeframe dataset into a lower timeframe dataset.

        This operation ensures that at each timestamp in the base dataset, only information
        from completed higher timeframe candles is available - preventing future sight/lookahead bias.

        Args:
            base_df: The lower timeframe DataFrame with DatetimeIndex
            higher_df: The higher timeframe DataFrame with DatetimeIndex

        Returns:
            A DataFrame with the same number of rows as base_df, containing features
            from the higher timeframe that were available at each base timeframe timestamp
        """
        pass


class MergeService(MergeServiceInterface):
    """
    Merges higher timeframe datasets into lower timeframe datasets.

    This service ensures that when merging data from multiple timeframes,
    only information that would have been available at each point in time
    is included, preventing lookahead bias in the resulting dataset.
    """

    def __init__(self) -> None:
        """Initialize the MergeService with logging."""
        self.logger = logging.getLogger(__name__)

    def merge_timeframes(
        self, base_df: pd.DataFrame, higher_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Performs an optimized two-pointer merge of higher timeframe data into base timeframe.

        For each row in the base dataframe, finds the last completed higher timeframe candle
        that would have been available at that timestamp, and merges its features.

        Args:
            base_df: The lower timeframe DataFrame with DatetimeIndex
            higher_df: The higher timeframe DataFrame with DatetimeIndex

        Returns:
            DataFrame with higher timeframe features merged into the base timeframe, preserving DatetimeIndex

        Note:
            - Columns from higher_df will be prefixed with 'HTF{minutes}_{column_name}'
            - OHLCV columns from higher_df are not included in the result
        """

        # Ensure both DataFrames have DateTimeIndex
        base_df = ensure_datetime_index(base_df, "base dataframe")

        # Check if higher_df is empty or lacks sufficient data
        if higher_df is None or len(higher_df) < 2:
            self.logger.warning(
                "Higher timeframe dataframe is empty or has insufficient data. Returning base dataframe with only the index."
            )
            # Return a DataFrame with the same index as base_df but no columns
            return pd.DataFrame(index=base_df.index)

        try:
            # Ensure higher_df has DateTimeIndex
            higher_df = ensure_datetime_index(higher_df, "higher timeframe dataframe")

            # Ensure data is properly sorted by index
            base_df = base_df.sort_index()
            higher_df = higher_df.sort_index()

            # Calculate the timeframe difference and create a label
            high_tf = detect_timeframe(higher_df)
            high_tf_label = int(high_tf.total_seconds() / 60)

            # Add close time as a column - this helps with merging logic
            higher_df["Close_Time"] = higher_df.index + high_tf

            # Prepare for two-pointer algorithm
            higher_idx = 0
            last_closed_candle = None
            merged_data = {}

            # Core merging algorithm - optimized for DataFrames with DatetimeIndex
            for timestamp, _row in base_df.iterrows():
                # Advance higher timeframe pointer as needed
                while (
                    higher_idx < len(higher_df)
                    and higher_df.iloc[higher_idx]["Close_Time"] <= timestamp
                ):
                    last_closed_candle = higher_df.iloc[higher_idx]
                    higher_idx += 1

                # Create merged row data
                merged_row = self._create_merged_row(
                    timestamp,
                    last_closed_candle,
                    higher_df.columns,
                    high_tf_label,
                )
                merged_data[timestamp] = (
                    merged_row  # Create DataFrame with DatetimeIndex
                )
            result_df = pd.DataFrame(merged_data).T

            # Preserve the index name from the base dataframe
            result_df.index.name = base_df.index.name
            return result_df

        except ValueError as e:
            # Handle errors from detect_timeframe gracefully
            self.logger.warning(
                f"Could not merge timeframes: {str(e)}. Returning base dataframe with only the index."
            )
            result_df = pd.DataFrame(index=base_df.index)
            # Preserve the index name from the base dataframe
            result_df.index.name = base_df.index.name
            return result_df

    def _create_merged_row(
        self,
        base_time: pd.Timestamp,
        higher_candle: Optional[pd.Series],
        higher_columns: pd.Index,
        high_tf_label: int,
    ) -> Dict:
        """
        Creates a merged data row combining base time with higher timeframe features.

        Args:
            base_time: The timestamp for the base timeframe row
            higher_candle: The last closed higher timeframe candle, or None if no candle is available
            higher_columns: The columns from the higher timeframe DataFrame
            high_tf_label: The numeric label for the higher timeframe (in minutes)

        Returns:
            Dictionary representing a merged row with higher timeframe features
        """
        merged_row = {}

        if higher_candle is not None:
            for col in higher_columns:
                if col not in [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Close_Time",
                    "Volume",
                ]:
                    merged_row[f"HTF-{high_tf_label}_{col}"] = higher_candle[col]

        return merged_row
