from pandas import DataFrame
from ai_trading.config.feature_config_collection import RviConfig
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature
import pandas_ta as ta

class RviFeature(BaseFeature):
    def __init__(self, source: DataFrame, postfix: str = ""):
        self.df_source = source
        self.postfix = postfix

    def compute(self, config: RviConfig) -> DataFrame:
        df = DataFrame()
        df["Time"] = self.df_source["Time"]
        df[f"rvi_{config.length}{self.postfix}"] = ta.rvi(
            self.df_source["Close"], self.df_source["High"], self.df_source["Low"], length=config.length
        )
        return df