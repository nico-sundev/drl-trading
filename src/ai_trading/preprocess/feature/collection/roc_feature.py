import pandas_ta as ta
from pandas import DataFrame

from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig
from ai_trading.config.feature_config_collection import RocConfig
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature


class RocFeature(BaseFeature):

    def __init__(
        self, source: DataFrame, config: BaseParameterSetConfig, postfix: str = ""
    ) -> None:
        super().__init__(source, config, postfix)
        self.config: RocConfig = self.config

    def compute(self) -> DataFrame:
        df = DataFrame()
        df["Time"] = self.df_source["Time"]
        df[f"roc_{self.config.length}{self.postfix}"] = ta.roc(
            self.df_source["Close"], length=self.config.length
        )
        return df

    def get_sub_features_names(self) -> list[str]:
        return [f"roc_{self.config.length}{self.postfix}"]
