import numpy as np
import pandas_ta as ta
from pandas import DataFrame

from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig
from ai_trading.config.feature_config_collection import MacdConfig
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature


class MacdFeature(BaseFeature):

    def __init__(
        self, source: DataFrame, config: BaseParameterSetConfig, postfix: str = ""
    ) -> None:
        super().__init__(source, config, postfix)
        self.config: MacdConfig = self.config

    def compute(self) -> DataFrame:
        df = DataFrame()
        df["Time"] = self.df_source["Time"]

        macd = ta.macd(
            self.df_source["Close"],
            fast=self.config.fast_length,
            slow=self.config.slow_length,
            signal=self.config.signal_length,
            fillna=np.nan,
            signal_indicators=True,
        )

        df["macd_cross_bullish" + self.postfix] = macd[
            f"MACDh_{self.config.fast_length}_{self.config.slow_length}_{self.config.signal_length}_XA_0"
        ]
        df["macd_cross_bearish" + self.postfix] = macd[
            f"MACDh_{self.config.fast_length}_{self.config.slow_length}_{self.config.signal_length}_XB_0"
        ]
        df["macd_trend" + self.postfix] = macd[
            f"MACD_{self.config.fast_length}_{self.config.slow_length}_{self.config.signal_length}_A_0"
        ]
        return df

    def get_sub_features_names(self) -> list[str]:
        return [
            f"macd_cross_bullish{self.postfix}",
            f"macd_cross_bearish{self.postfix}",
            f"macd_trend{self.postfix}",
        ]
