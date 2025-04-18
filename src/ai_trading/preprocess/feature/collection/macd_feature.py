from typing import cast

import pandas_ta as ta  # type: ignore
from pandas import DataFrame

from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig
from ai_trading.config.feature_config_collection import MacdConfig
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature


class MacdFeature(BaseFeature):
    def __init__(self, source: DataFrame, postfix: str = "") -> None:
        self.df_source = source
        self.postfix = postfix

    def compute(self, config: BaseParameterSetConfig) -> DataFrame:
        macd_config = cast(MacdConfig, config)
        df = DataFrame()
        df["Time"] = self.df_source["Time"]

        macd = ta.macd(
            self.df_source["Close"],
            fast=macd_config.fast,
            slow=macd_config.slow,
            signal=macd_config.signal,
        )

        df["macd_cross_bullish" + self.postfix] = macd[
            f"MACDh_{macd_config.fast_length}_{macd_config.slow_length}_{macd_config.signal_length}_XA_0"
        ]
        df["macd_cross_bearish" + self.postfix] = macd[
            f"MACDh_{macd_config.fast_length}_{macd_config.slow_length}_{macd_config.signal_length}_XB_0"
        ]
        df["macd_trend" + self.postfix] = macd[
            f"MACD_{macd_config.fast_length}_{macd_config.slow_length}_{macd_config.signal_length}_A_0"
        ]
        return df

    def get_sub_features_names(self, config: MacdConfig) -> list[str]:
        return [
            f"macd_cross_bullish{self.postfix}",
            f"macd_cross_bearish{self.postfix}",
            f"macd_trend{self.postfix}",
        ]
