from typing import cast

import pandas_ta as ta
from pandas import DataFrame

from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig
from ai_trading.config.feature_config_collection import RocConfig
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature


class RocFeature(BaseFeature):
    def __init__(self, source: DataFrame, postfix: str = "") -> None:
        self.df_source = source
        self.postfix = postfix

    def compute(self, config: BaseParameterSetConfig) -> DataFrame:
        roc_config = cast(RocConfig, config)
        df = DataFrame()
        df["Time"] = self.df_source["Time"]
        df[f"roc_{roc_config.length}{self.postfix}"] = ta.roc(
            self.df_source["Close"], length=roc_config.length
        )
        return df

    def get_sub_features_names(self, config: BaseParameterSetConfig) -> list[str]:
        roc_config = cast(RocConfig, config)
        return [f"roc_{roc_config.length}{self.postfix}"]
