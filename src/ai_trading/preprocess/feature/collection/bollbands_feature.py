from pandas import DataFrame
from ai_trading.config.feature_config_collection import BollbandsConfig
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature
import pandas_ta as ta

class BollbandsFeature(BaseFeature):
    def __init__(self, source: DataFrame, postfix: str = ""):
        self.df_source = source
        self.postfix = postfix

    def compute(self, config: BollbandsConfig) -> DataFrame:
        df = DataFrame()
        df["Time"] = self.df_source["Time"]
        # TODO: implement bollbands logic here
        return df

    def get_sub_features_names(self, config: BollbandsConfig) -> list[str]:
        return [
            f"bollinger_upper{config.length}{self.postfix}",
            f"bollinger_middle{config.length}{self.postfix}",
            f"bollinger_lower{config.length}{self.postfix}"
        ]
