import numpy as np
from pandas import DataFrame
from ai_trading.config.feature_config_collection import MacdConfig
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature
import pandas_ta as ta

class MacdFeature(BaseFeature):
    def __init__(self, source: DataFrame, postfix: str = ""):
        self.df_source = source
        self.postfix = postfix

    def compute(self, config: MacdConfig) -> DataFrame:
        macd = ta.macd(
            self.df_source["Close"],
            fast=config.fast,
            slow=config.slow,
            signal=config.signal,
            fillna=np.nan,
            signal_indicators=True,
        )

        df = DataFrame()
        df["Time"] = self.df_source["Time"]
        df["macd_cross_bullish" + self.postfix] = macd[f"MACDh_{config.fast}_{config.slow}_{config.signal}_XA_0"]
        df["macd_cross_bearish" + self.postfix] = macd[f"MACDh_{config.fast}_{config.slow}_{config.signal}_XB_0"]
        df["macd_trend" + self.postfix] = macd[f"MACD_{config.fast}_{config.slow}_{config.signal}_A_0"]
        return df