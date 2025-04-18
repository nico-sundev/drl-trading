from typing import cast

import pandas_ta as ta
from pandas import DataFrame

from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig
from ai_trading.config.feature_config_collection import RviConfig
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature


class RviFeature(BaseFeature):
    def __init__(self, source: DataFrame, postfix: str = "") -> None:
        self.df_source = source
        self.postfix = postfix

    def compute(self, config: BaseParameterSetConfig) -> DataFrame:
        rvi_config = cast(RviConfig, config)
        df = DataFrame()
        df["Time"] = self.df_source["Time"]
        df[f"rvi_{rvi_config.length}{self.postfix}"] = ta.rvi(
            self.df_source["Close"],
            self.df_source["High"],
            self.df_source["Low"],
            length=rvi_config.length,
        )
        return df

    def get_sub_features_names(self, config: BaseParameterSetConfig) -> list[str]:
        rvi_config = cast(RviConfig, config)
        return [f"rvi_{rvi_config.length}{self.postfix}"]
