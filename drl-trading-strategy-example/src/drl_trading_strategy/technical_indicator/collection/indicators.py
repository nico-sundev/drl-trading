from typing import Optional

from drl_trading_common.base.base_indicator import BaseIndicator
from drl_trading_strategy.decorator.indicator_type_decorator import indicator_type
from drl_trading_strategy.enum.indicator_type_enum import IndicatorTypeEnum
from drl_trading_strategy.feature.config.feature_configs import RsiConfig
from pandas import DataFrame
from talipp.indicators import RSI


@indicator_type(IndicatorTypeEnum.RSI)
class RsiIndicator(BaseIndicator):
    def __init__(self, config: RsiConfig):
        self.config: RsiConfig = config
        self.indicator: RSI = RSI(period=self.config.length)

    def get_all(self) -> Optional[DataFrame]:
        if len(self.indicator) == 0:
            raise ValueError("RSI indicator has not been computed yet.")

        return DataFrame({
            "rsi": [self.indicator]
        })

    def add(self, value: DataFrame) -> None:
        self.indicator.add(value["Close"].values)

    def get_latest(self) -> Optional[DataFrame]:
        if len(self.indicator) == 0:
            raise ValueError("RSI indicator has not been computed yet.")

        return DataFrame({
            "rsi": [self.indicator[-1]]
        })
