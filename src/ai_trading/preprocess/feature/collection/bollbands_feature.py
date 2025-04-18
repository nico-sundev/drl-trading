from typing import cast

from pandas import DataFrame

from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig
from ai_trading.config.feature_config_collection import BollbandsConfig
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature


class BollbandsFeature(BaseFeature):
    def __init__(self, source: DataFrame, postfix: str = "") -> None:
        self.df_source = source
        self.postfix = postfix

    def compute(self, config: BaseParameterSetConfig) -> DataFrame:
        bollbands_config = cast(BollbandsConfig, config)
        df = DataFrame()
        df["Time"] = self.df_source["Time"]

        sma = self.df_source["Close"].rolling(window=bollbands_config.length).mean()
        std = self.df_source["Close"].rolling(window=bollbands_config.length).std()

        df[f"bb_upper{self.postfix}"] = sma + bollbands_config.std_dev * std
        df[f"bb_middle{self.postfix}"] = sma
        df[f"bb_lower{self.postfix}"] = sma - bollbands_config.std_dev * std

        return df

    def get_sub_features_names(self, config: BaseParameterSetConfig) -> list[str]:
        return [
            f"bb_upper{self.postfix}",
            f"bb_middle{self.postfix}",
            f"bb_lower{self.postfix}",
        ]
