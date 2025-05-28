from typing import Optional

import pandas_ta as ta
from drl_trading_framework.preprocess.feature.collection.base_feature import BaseFeature
from drl_trading_framework.preprocess.metrics.technical_metrics_service import (
    TechnicalMetricsServiceInterface,
)
from pandas import DataFrame

from drl_trading_common.config.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_common.config.feature_config_collection import RsiConfig


class RsiFeature(BaseFeature):

    def __init__(
        self,
        source: DataFrame,
        config: BaseParameterSetConfig,
        postfix: str = "",
        metrics_service: Optional[TechnicalMetricsServiceInterface] = None,
    ) -> None:
        super().__init__(source, config, postfix, metrics_service)
        self.config: RsiConfig = self.config

    def compute(self) -> DataFrame:
        # Get source DataFrame with ensured DatetimeIndex using the base class method
        source_df = self._prepare_source_df()

        # Create a DataFrame with the same index as the source
        rsi_values = ta.rsi(source_df["Close"], length=self.config.length)

        # Create result DataFrame with both Time column and feature values
        df = DataFrame(index=source_df.index)
        df[f"rsi_{self.config.length}{self.postfix}"] = rsi_values

        return df

    def get_sub_features_names(self) -> list[str]:
        return [f"rsi_{self.config.length}{self.postfix}"]
