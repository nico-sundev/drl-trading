import abc
import logging
from typing import List

import pandas as pd

from ai_trading.data_set_utils.util import (
    detect_timeframe,
    separate_asset_price_datasets,
)
from ai_trading.model.asset_price_dataset import AssetPriceDataSet


class StripServiceInterface(abc.ABC):
    """
    Interface for services that strip higher timeframe datasets based on base dataset boundaries.

    This interface ensures consistent behavior across implementations that handle trimming
    of higher timeframe data to align with lower timeframe (base) datasets.
    """

    @abc.abstractmethod
    def strip_higher_timeframes(
        self,
        base_start_timestamp: pd.Timestamp,
        base_end_timestamp: pd.Timestamp,
        higher_timeframe_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Strips the higher timeframe dataframe at both ends to ensure it aligns with base dataset boundaries.

        The method ensures that higher timeframe data:
        1. Doesn't start too early (before base_start - 2*timeframe_duration)
        2. Doesn't end too late (after base_end + 2*timeframe_duration)

        Args:
            base_start_timestamp: The start timestamp of the base dataset
            base_end_timestamp: The end timestamp of the base dataset
            higher_timeframe_df: The dataframe of the higher timeframe to be stripped

        Returns:
            DataFrame with rows filtered to the appropriate time range

        Raises:
            ValueError: If the input DataFrame doesn't contain proper timestamp data
        """
        pass

    @abc.abstractmethod
    def strip_asset_price_datasets(
        self,
        datasets: List[AssetPriceDataSet],
    ) -> List[AssetPriceDataSet]:
        """
        Strips all higher timeframe datasets based on the base dataset's time range.

        This method processes a list of datasets, identifies the base dataset,
        and strips all other datasets to align with the base dataset's time boundaries.

        Args:
            datasets: List of asset price datasets with one base dataset and
                      multiple higher timeframe datasets

        Returns:
            List of asset price datasets with the higher timeframe datasets properly stripped

        Raises:
            ValueError: If no base dataset is found or if datasets cannot be properly processed
        """
        pass


class StripService(StripServiceInterface):
    """
    Service to strip higher timeframe datasets based on the start and end timestamps of the base dataset.

    This implementation ensures proper alignment between base datasets and higher timeframe
    datasets by removing excess data at the beginning and end of higher timeframe datasets.
    """

    def __init__(self) -> None:
        """Initialize the StripService with logging capability."""
        self._logger = logging.getLogger(__name__)

    def strip_higher_timeframes(
        self,
        base_start_timestamp: pd.Timestamp,
        base_end_timestamp: pd.Timestamp,
        higher_timeframe_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Strips the higher timeframe dataframe at both ends to ensure:
        1. No timestamps exceed the base dataset's end timestamp + 2*timeframe_duration
        2. No timestamps precede the base dataset's start timestamp - 2*timeframe_duration

        Args:
            base_start_timestamp: The start timestamp of the base dataset
            base_end_timestamp: The end timestamp of the base dataset
            higher_timeframe_df: The dataframe of the higher timeframe

        Returns:
            DataFrame: The stripped dataframe

        Raises:
            ValueError: If the dataframe cannot be properly processed or if timeframe detection fails
        """
        # Calculate the timeframe duration
        timeframe_duration = detect_timeframe(higher_timeframe_df)

        # Calculate threshold timestamps for beginning and end
        begin_threshold_timestamp = base_start_timestamp - 2 * timeframe_duration
        end_threshold_timestamp = base_end_timestamp + 2 * timeframe_duration

        self._logger.info(
            f"Stripping higher timeframe dataframe. Base start: {base_start_timestamp}, Base end: {base_end_timestamp}, "
            f"Timeframe duration: {timeframe_duration}, Begin threshold: {begin_threshold_timestamp}, "
            f"End threshold: {end_threshold_timestamp}"
        )

        # Filter the dataframe to include only rows between the thresholds
        stripped_df: pd.DataFrame = higher_timeframe_df[
            (higher_timeframe_df["Time"] >= begin_threshold_timestamp)
            & (higher_timeframe_df["Time"] < end_threshold_timestamp)
        ]

        # Log results
        original_count = len(higher_timeframe_df)
        stripped_count = len(stripped_df)

        if stripped_count > 0:
            first_row = stripped_df.iloc[0]
            last_row = stripped_df.iloc[-1]
            self._logger.info(
                f"Stripping complete. Removed {original_count - stripped_count} rows. "
                f"First row after strip: {first_row['Time']}, Last row after strip: {last_row['Time']}"
            )
        else:
            self._logger.warning("Stripping resulted in an empty dataframe.")

        return stripped_df

    def strip_asset_price_datasets(
        self,
        datasets: List[AssetPriceDataSet],
    ) -> List[AssetPriceDataSet]:
        """
        Strips all higher timeframe datasets based on the base dataset's time range.

        Args:
            datasets: List of asset price datasets with one base dataset and multiple higher timeframe datasets

        Returns:
            List of asset price datasets with the higher timeframe datasets properly stripped

        Raises:
            ValueError: If no base dataset is found in the input datasets
        """
        base_dataset, other_datasets = separate_asset_price_datasets(datasets)

        # Get base dataset time boundaries
        base_start_timestamp = base_dataset.asset_price_dataset["Time"].iloc[0]
        base_end_timestamp = base_dataset.asset_price_dataset["Time"].iloc[-1]

        stripped_other_datasets = [
            AssetPriceDataSet(
                timeframe=dataset.timeframe,
                base_dataset=dataset.base_dataset,
                asset_price_dataset=self.strip_higher_timeframes(
                    base_start_timestamp,
                    base_end_timestamp,
                    dataset.asset_price_dataset,
                ),
            )
            for dataset in other_datasets
        ]
        return [base_dataset, *stripped_other_datasets]
