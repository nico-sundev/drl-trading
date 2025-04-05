import numpy as np
from pandas import DataFrame
from ai_trading.config.feature_config_mapper import RsiConfig
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature
import pandas_ta as ta

class RsiFeature(BaseFeature):
    def __init__(self, source: DataFrame, postfix: str = ""):
        self.df_source = source
        self.postfix = postfix

    def compute(self, config: RsiConfig) -> DataFrame:
        df = DataFrame()
        df["Time"] = self.df_source["Time"]
        df[f"rsi_{config.length}{self.postfix}"] = ta.rsi(
            self.df_source["Close"], length=config.length
        )
        return df
