import logging

import pandas as pd

from ai_trading.data_set_utils.util import (
    detect_timeframe,
    separate_asset_price_datasets,
)
from ai_trading.model.asset_price_dataset import AssetPriceDataSet


class TimeframeStripperService:
    """
    Service to strip higher timeframe datasets based on the end timestamp of the base dataset.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def strip_higher_timeframes(
        self, base_end_timestamp: pd.Timestamp, higher_timeframe_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Strips the higher timeframe dataframe to ensure no timestamps exceed the base dataset's end timestamp + its timeframe duration.

        Args:
            base_end_timestamp (pd.Timestamp): The end timestamp of the base dataset.
            higher_timeframe_df (pd.DataFrame): The dataframe of the higher timeframe.

        Returns:
            pd.DataFrame: The stripped dataframe.
        """
        # Calculate the threshold timestamp for the higher timeframe
        timeframe_duration = detect_timeframe(higher_timeframe_df)
        threshold_timestamp = base_end_timestamp + 2 * timeframe_duration

        self.logger.info(
            f"Stripping higher timeframe dataframe. Base end timestamp: {base_end_timestamp}, "
            f"Timeframe duration: {timeframe_duration}, Threshold timestamp: {threshold_timestamp}"
        )

        # Filter the dataframe to include only rows where the open timestamp is below the threshold
        stripped_df = higher_timeframe_df[
            higher_timeframe_df["Time"] < threshold_timestamp
        ]

        if not stripped_df.empty:
            last_row = stripped_df.iloc[-1]
            self.logger.info(
                f"Stripping complete. Last row after strip: {last_row.to_dict()}"
            )
        else:
            self.logger.warning("Stripping resulted in an empty dataframe.")

        return stripped_df

    def strip_asset_price_datasets(
        self,
        datasets: list[AssetPriceDataSet],
    ) -> list[AssetPriceDataSet]:
        base_dataset, other_datasets = separate_asset_price_datasets(datasets)

        stripped_other_datasets = [
            AssetPriceDataSet(
                timeframe=dataset.timeframe,
                base_dataset=dataset.base_dataset,
                asset_price_dataset=self.strip_higher_timeframes(
                    base_dataset.asset_price_dataset["Time"].iloc[-1],
                    dataset.asset_price_dataset,
                ),
            )
            for dataset in other_datasets
        ]
        return [base_dataset, *stripped_other_datasets]
