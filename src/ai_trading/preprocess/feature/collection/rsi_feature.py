import pandas_ta as ta
from pandas import DataFrame

from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig
from ai_trading.config.feature_config_collection import RsiConfig
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature


class RsiFeature(BaseFeature):

    def __init__(
        self, source: DataFrame, config: BaseParameterSetConfig, postfix: str = ""
    ) -> None:
        super().__init__(source, config, postfix)
        self.config: RsiConfig = self.config

    def compute(self) -> DataFrame:
        df = DataFrame()
        df["Time"] = self.df_source["Time"]
        df[f"rsi_{self.config.length}{self.postfix}"] = ta.rsi(
            self.df_source["Close"], length=self.config.length
        )
        return df

    def get_sub_features_names(self) -> list[str]:
        return [f"rsi_{self.config.length}{self.postfix}"]
