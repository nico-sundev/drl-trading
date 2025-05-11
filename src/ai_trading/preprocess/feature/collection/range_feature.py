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
        # Get source DataFrame with ensured DatetimeIndex using the base class method
        source_df = self._prepare_source_df()

        # Create result DataFrame with the same index
        df = DataFrame(index=source_df.index)

        # Create the SupportResistanceFinder with the prepared source DataFrame
        finder = SupportResistanceFinder(
            source_df, self.config.lookback, self.config.wick_handle_strategy
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
