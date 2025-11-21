from typing import Optional

from pandas import DataFrame
from talipp.indicators import RSI

from drl_trading_common.base.base_indicator import BaseIndicator
from drl_trading_strategy_example.decorator.indicator_type_decorator import (
    indicator_type,
)
from drl_trading_strategy_example.enum.indicator_type_enum import IndicatorTypeEnum
from drl_trading_strategy_example.feature.config.feature_configs import RsiConfig


@indicator_type(IndicatorTypeEnum.RSI)
class RsiIndicator(BaseIndicator):
    """RSI indicator that preserves timestamp information alongside indicator values."""

    def __init__(self, config: RsiConfig):
        super().__init__()
        self.config: RsiConfig = config
        self.indicator: RSI = RSI(period=self.config.length)

    def get_all(self) -> Optional[DataFrame]:
        """
        Get all RSI values with their corresponding timestamps.

        Returns:
            DataFrame with DatetimeIndex containing all RSI values

        Raises:
            ValueError: If indicator has not been computed yet
        """
        if len(self.indicator) == 0:
            raise ValueError("RSI indicator has not been computed yet.")

        return self._create_result_dataframe(self.indicator.output_values, "rsi")

    def add(self, value: DataFrame) -> None:
        """
        Add new OHLCV data to compute RSI values.

        Extracts Close prices and timestamps from the input DataFrame.
        Preserves the timestamp information for proper alignment.

        Args:
            value: DataFrame with Close prices and DatetimeIndex
        """
        self._store_timestamps(value)
        self.indicator.add(value["Close"].values.tolist())

    def get_latest(self) -> Optional[DataFrame]:
        """
        Get latest RSI value with its corresponding timestamp.

        Returns:
            DataFrame with DatetimeIndex containing latest RSI value

        Raises:
            ValueError: If indicator has not been computed yet
        """
        if len(self.indicator) == 0:
            raise ValueError("RSI indicator has not been computed yet.")

        latest_timestamp_idx = len(self.indicator) - 1
        latest_timestamp = self.timestamps[latest_timestamp_idx]

        return DataFrame({"rsi": [self.indicator[-1]]}, index=[latest_timestamp])
