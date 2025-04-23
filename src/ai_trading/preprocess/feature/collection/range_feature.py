from pandas import DataFrame

from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig
from ai_trading.config.feature_config_collection import RangeConfig
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature
from ai_trading.preprocess.feature.custom.range_indicator import SupportResistanceFinder


class RangeFeature(BaseFeature):

    def __init__(
        self, source: DataFrame, config: BaseParameterSetConfig, postfix: str = ""
    ) -> None:
        super().__init__(source, config, postfix)
        self.config: RangeConfig = self.config

    def compute(self) -> DataFrame:
        df = DataFrame()
        df["Time"] = self.df_source["Time"]
        finder = SupportResistanceFinder(
            self.df_source, self.config.lookback, self.config.wick_handle_strategy
        )
        ranges = finder.find_support_resistance_zones()
        df[f"resistance_range{self.config.lookback}{self.postfix}"] = ranges[
            "resistance_range"
        ]
        df[f"support_range{self.config.lookback}{self.postfix}"] = ranges[
            "support_range"
        ]
        return df

    def get_sub_features_names(self) -> list[str]:
        return [
            f"resistance_range{self.config.lookback}{self.postfix}",
            f"support_range{self.config.lookback}{self.postfix}",
        ]
