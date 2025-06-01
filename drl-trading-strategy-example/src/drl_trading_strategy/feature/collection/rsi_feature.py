
import pandas_ta as ta
from drl_trading_common import BaseParameterSetConfig
from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_core.preprocess.feature.technical_indicators_service import (
    TechnicalIndicatorService,
)
from drl_trading_strategy.feature.config import RsiConfig
from pandas import DataFrame


class RsiFeature(BaseFeature):

    def __init__(
        self,
        source: DataFrame,
        config: BaseParameterSetConfig,
        indicator_service: TechnicalIndicatorService,
        postfix: str = "",
    ) -> None:
        super().__init__(source, config, postfix, indicator_service)
        self.config: RsiConfig = self.config
        self.indicator_name = f"rsi_{self.config.length}{self.postfix}"
        self.indicator_service.register_instance(self.indicator_name, "rsi", period=config.period)

    def update(self, record) -> float:
        self.indicator_service.update(self.indicator_name, record)
        val = self.indicator_service.latest(self.indicator_name)
        # if self.config.use_slope:
        #     return compute_slope(self.indicator_service.series(self.indicator_name), window=self.config.slope_window)
        return val
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
