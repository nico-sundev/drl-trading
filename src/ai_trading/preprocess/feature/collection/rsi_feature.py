from typing import cast

import pandas_ta as ta
from pandas import DataFrame

from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig
from ai_trading.config.feature_config_collection import RsiConfig
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature


class RsiFeature(BaseFeature):
    def __init__(self, source: DataFrame, postfix: str = "") -> None:
        self.df_source = source
        self.postfix = postfix

    def compute(self, config: BaseParameterSetConfig) -> DataFrame:
        rsi_config = cast(RsiConfig, config)
        df = DataFrame()
        df["Time"] = self.df_source["Time"]
        df[f"rsi_{rsi_config.length}{self.postfix}"] = ta.rsi(
            self.df_source["Close"], length=rsi_config.length
        )
        return df

    def get_sub_features_names(self, config: BaseParameterSetConfig) -> list[str]:
        rsi_config = cast(RsiConfig, config)
        return [f"rsi_{rsi_config.length}{self.postfix}"]
