
from typing import Optional

from drl_trading_common import BaseParameterSetConfig
from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.decorator.feature_role_decorator import feature_role
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.interface.indicator.technical_indicator_facade_interface import (
    ITechnicalIndicatorFacade,
)
from drl_trading_common.model.dataset_identifier import DatasetIdentifier
from drl_trading_strategy_example.decorator import feature_type
from drl_trading_strategy_example.decorator.feature_type_decorator import (
    get_feature_type_from_class,
)
from drl_trading_strategy_example.enum.feature_type_enum import FeatureTypeEnum
from drl_trading_strategy_example.feature.config import RsiConfig
from drl_trading_strategy_example.mapper.mapper import TypeMapper
from pandas import DataFrame


@feature_type(FeatureTypeEnum.RSI)
@feature_role(FeatureRoleEnum.OBSERVATION_SPACE)
class RsiFeature(BaseFeature):

    def __init__(
        self,
        config: BaseParameterSetConfig,
        dataset_id: DatasetIdentifier,
        indicator_service: ITechnicalIndicatorFacade,
        postfix: str = ""
    ) -> None:
        super().__init__(config, dataset_id, indicator_service, postfix)
        self.config: RsiConfig = self.config
        self.feature_name = f"rsi_{self.config.length}{self.postfix}"
        self.indicator_service.register_instance(self.feature_name, self._get_indicator_type(), period=self.config.length)

    def add(self, df: DataFrame) -> None:
        index_corrected_dataframe = self._prepare_source_df(df)
        self.indicator_service.add(self.feature_name, index_corrected_dataframe)

    def compute_latest(self) -> Optional[DataFrame]:
        return self.indicator_service.get_latest(self.feature_name)

    def compute_all(self) -> Optional[DataFrame]:
        # Create a DataFrame with the same index as the source
        rsi_values = self.indicator_service.get_all(self.feature_name)
        # Create result DataFrame with both Time column and feature values
        df = DataFrame()
        df[f"rsi_{self.config.length}{self.postfix}"] = rsi_values
        return df

    def get_sub_features_names(self) -> list[str]:
        return ["value"]

    def _get_feature_type(self) -> FeatureTypeEnum:
        return get_feature_type_from_class(self.__class__)

    def _get_indicator_type(self) -> str:
        """Get the indicator type as a string for decoupling from enums."""
        enum_type = TypeMapper.map_feature_to_indicator(self._get_feature_type())
        return enum_type.value

    def get_feature_name(self) -> str:
        return self._get_feature_type().value

    def get_config_to_string(self) -> str:
        return f"{self.config.length}"
