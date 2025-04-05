from pandas import DataFrame
from ai_trading.config.feature_config_mapper import RocConfig
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature
import pandas_ta as ta

class RocFeature(BaseFeature):
    def __init__(self, source: DataFrame, postfix: str = ""):
        self.df_source = source
        self.postfix = postfix

    def compute(self, config: RocConfig) -> DataFrame:
        df = DataFrame()
        df["Time"] = self.df_source["Time"]
        df[f"roc_{config.length}{self.postfix}"] = ta.roc(
            self.df_source["Close"], length=config.length
        )
        return df