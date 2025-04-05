from pandas import DataFrame
from ai_trading.config.feature_config_mapper import RangeConfig
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature
import pandas_ta as ta

from ai_trading.preprocess.feature.custom.range_indicator import SupportResistanceFinder

class RangeFeature(BaseFeature):
    def __init__(self, source: DataFrame, postfix: str = ""):
        self.df_source = source
        self.postfix = postfix

    def compute(self, config: RangeConfig) -> DataFrame:
        df = DataFrame()
        df["Time"] = self.df_source["Time"]
        finder = SupportResistanceFinder(
            self.df_source, config.lookback, config.wick_handle_strategy
        )
        ranges = finder.find_support_resistance_zones()
        df[f"resistance_range{config.lookback}{self.postfix}"] = ranges["resistance_range"]
        df[f"support_range{config.lookback}{self.postfix}"] = ranges["support_range"]
        return df