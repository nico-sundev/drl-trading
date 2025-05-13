from typing import Optional

from pandas import DataFrame

from drl_trading_framework.common.config.base_parameter_set_config import (
    BaseParameterSetConfig,
)
from drl_trading_framework.common.config.feature_config_collection import RangeConfig
from drl_trading_framework.preprocess.feature.collection.base_feature import BaseFeature
from drl_trading_framework.preprocess.feature.custom.range_indicator import (
    SupportResistanceFinder,
)
from drl_trading_framework.preprocess.metrics.technical_metrics_service import (
    TechnicalMetricsServiceInterface,
)


class RangeFeature(BaseFeature):

    def __init__(
        self,
        source: DataFrame,
        config: BaseParameterSetConfig,
        postfix: str = "",
        metrics_service: Optional[TechnicalMetricsServiceInterface] = None,
    ) -> None:
        super().__init__(source, config, postfix, metrics_service)
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
