from typing import Optional

from pandas import DataFrame

from ai_trading.common.config.base_parameter_set_config import BaseParameterSetConfig
from ai_trading.common.config.feature_config_collection import BollbandsConfig
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature
from ai_trading.preprocess.metrics.technical_metrics_service import (
    TechnicalMetricsServiceInterface,
)


class BollbandsFeature(BaseFeature):

    def __init__(
        self,
        source: DataFrame,
        config: BaseParameterSetConfig,
        postfix: str = "",
        metrics_service: Optional[TechnicalMetricsServiceInterface] = None,
    ) -> None:
        super().__init__(source, config, postfix, metrics_service)
        self.config: BollbandsConfig = self.config

    def compute(self) -> DataFrame:
        # Get source DataFrame with ensured DatetimeIndex using the base class method
        source_df = self._prepare_source_df()

        # Create a DataFrame with the same index
        df = DataFrame(index=source_df.index)

        # Calculate SMA and standard deviation
        sma = source_df["Close"].rolling(window=self.config.length).mean()
        std = source_df["Close"].rolling(window=self.config.length).std()

        # Standard Bollinger Bands
        df[f"bb_upper{self.postfix}"] = sma + self.config.std_dev * std
        df[f"bb_middle{self.postfix}"] = sma
        df[f"bb_lower{self.postfix}"] = sma - self.config.std_dev * std

        # Add ATR-adjusted bands if metrics service is available
        if self.metrics_service is not None:
            atr_df = self.metrics_service.get_atr(period=self.config.length)
            if atr_df is not None:
                # Create ATR-normalized bands that adjust for volatility
                df[f"bb_atr_upper{self.postfix}"] = (
                    sma + self.config.std_dev * atr_df["ATR"]
                )
                df[f"bb_atr_lower{self.postfix}"] = (
                    sma - self.config.std_dev * atr_df["ATR"]
                )
                df[f"bb_atr_width{self.postfix}"] = (
                    df[f"bb_atr_upper{self.postfix}"]
                    - df[f"bb_atr_lower{self.postfix}"]
                ) / sma

        return df

    def get_sub_features_names(self) -> list[str]:
        base_features = [
            f"bb_upper{self.postfix}",
            f"bb_middle{self.postfix}",
            f"bb_lower{self.postfix}",
        ]

        # Add ATR-related features if available
        if self.metrics_service is not None:
            atr_features = [
                f"bb_atr_upper{self.postfix}",
                f"bb_atr_lower{self.postfix}",
                f"bb_atr_width{self.postfix}",
            ]
            return base_features + atr_features

        return base_features
