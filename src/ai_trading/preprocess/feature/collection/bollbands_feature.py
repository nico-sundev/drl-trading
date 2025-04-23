from pandas import DataFrame

from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig
from ai_trading.config.feature_config_collection import BollbandsConfig
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature


class BollbandsFeature(BaseFeature):

    def __init__(
        self, source: DataFrame, config: BaseParameterSetConfig, postfix: str = ""
    ) -> None:
        super().__init__(source, config, postfix)
        self.config: BollbandsConfig = self.config

    def compute(self) -> DataFrame:
        df = DataFrame()
        df["Time"] = self.df_source["Time"]

        sma = self.df_source["Close"].rolling(window=self.config.length).mean()
        std = self.df_source["Close"].rolling(window=self.config.length).std()

        df[f"bb_upper{self.postfix}"] = sma + self.config.std_dev * std
        df[f"bb_middle{self.postfix}"] = sma
        df[f"bb_lower{self.postfix}"] = sma - self.config.std_dev * std

        return df

    def get_sub_features_names(self) -> list[str]:
        return [
            f"bb_upper{self.postfix}",
            f"bb_middle{self.postfix}",
            f"bb_lower{self.postfix}",
        ]
