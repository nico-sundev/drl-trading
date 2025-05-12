from typing import Optional

import numpy as np
import pandas_ta as ta
from pandas import DataFrame

from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig
from ai_trading.config.feature_config_collection import MacdConfig
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature
from ai_trading.preprocess.metrics.technical_metrics_service import (
    TechnicalMetricsServiceInterface,
)


class MacdFeature(BaseFeature):

    def __init__(
        self,
        source: DataFrame,
        config: BaseParameterSetConfig,
        postfix: str = "",
        metrics_service: Optional[TechnicalMetricsServiceInterface] = None,
    ) -> None:
        super().__init__(source, config, postfix, metrics_service)
        self.config: MacdConfig = self.config

    def compute(self) -> DataFrame:
        # Get source DataFrame with ensured DatetimeIndex using the base class method
        source_df = self._prepare_source_df()

        # Create a DataFrame with the same index
        df = DataFrame(index=source_df.index)

        macd = ta.macd(
            source_df["Close"],
            fast=self.config.fast_length,
            slow=self.config.slow_length,
            signal=self.config.signal_length,
            fillna=np.nan,
            signal_indicators=True,
        )

        df["macd_cross_bullish" + self.postfix] = macd[
            f"MACDh_{self.config.fast_length}_{self.config.slow_length}_{self.config.signal_length}_XA_0"
        ]
        df["macd_cross_bearish" + self.postfix] = macd[
            f"MACDh_{self.config.fast_length}_{self.config.slow_length}_{self.config.signal_length}_XB_0"
        ]
        df["macd_trend" + self.postfix] = macd[
            f"MACD_{self.config.fast_length}_{self.config.slow_length}_{self.config.signal_length}_A_0"
        ]

        return df

    def get_sub_features_names(self) -> list[str]:
        return [
            f"macd_cross_bullish{self.postfix}",
            f"macd_cross_bearish{self.postfix}",
            f"macd_trend{self.postfix}",
        ]
