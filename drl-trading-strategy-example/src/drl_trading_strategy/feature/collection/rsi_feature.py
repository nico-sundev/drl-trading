
from typing import Optional

from drl_trading_common import BaseParameterSetConfig
from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.interfaces.technical_indicator_service_interface import (
    TechnicalIndicatorFacadeInterface,
)
from drl_trading_strategy.decorators import feature_type
from drl_trading_strategy.decorators.feature_type_decorator import (
    get_feature_type_from_class,
)
from drl_trading_strategy.enum.feature_type_enum import FeatureTypeEnum
from drl_trading_strategy.enum.indicator_type_enum import IndicatorTypeEnum
from drl_trading_strategy.feature.config import RsiConfig
from drl_trading_strategy.mapper.mapper import TypeMapper
from pandas import DataFrame


@feature_type(FeatureTypeEnum.RSI)
class RsiFeature(BaseFeature):

    def __init__(
        self,
        source: DataFrame,
        config: BaseParameterSetConfig,
        indicator_service: TechnicalIndicatorFacadeInterface,
        postfix: str = "",
    ) -> None:
        super().__init__(source, config, indicator_service, postfix)
        self.config: RsiConfig = self.config
        self.feature_name = f"rsi_{self.config.length}{self.postfix}"
        self.indicator_service.register_instance(self.feature_name, self._get_indicator_type(), period=self.config.length)

    def add(self, df: DataFrame) -> None:
        self.indicator_service.add(self.feature_name, df)

    def compute_latest(self) -> Optional[DataFrame]:
        return self.indicator_service.get_latest(self.feature_name)

    def compute_all(self) -> Optional[DataFrame]:
        source_df = self._prepare_source_df()
        self.indicator_service.add(self.feature_name, source_df)

        # Create a DataFrame with the same index as the source
        rsi_values = self.indicator_service.get_all(self.feature_name)
        # Create result DataFrame with both Time column and feature values
        df = DataFrame(index=source_df.index)
        df[f"rsi_{self.config.length}{self.postfix}"] = rsi_values
        return df

    def get_sub_features_names(self) -> list[str]:
        return [f"rsi_{self.config.length}{self.postfix}"]

    def _get_feature_type(self) -> FeatureTypeEnum:
        return get_feature_type_from_class(self.__class__)

    def _get_indicator_type(self) -> IndicatorTypeEnum:
        return TypeMapper.map_feature_to_indicator(self._get_feature_type())

    def get_feature_name(self) -> str:
        return self._get_feature_type().value
