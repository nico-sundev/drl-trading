
from typing import Optional

from drl_trading_common import BaseParameterSetConfig
from drl_trading_common.core.model.base_feature import BaseFeature
from drl_trading_common.decorator.feature_role_decorator import feature_role
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.interface.indicator.technical_indicator_facade_interface import (
    ITechnicalIndicatorFacade,
)
from drl_trading_common.adapter.model.dataset_identifier import DatasetIdentifier
from drl_trading_strategy_example.decorator import feature_type
from drl_trading_strategy_example.decorator.feature_type_decorator import (
    get_feature_type_from_class,
)
from drl_trading_strategy_example.enum.feature_type_enum import FeatureTypeEnum
from pandas import DataFrame


@feature_type(FeatureTypeEnum.CLOSE_PRICE)
@feature_role(FeatureRoleEnum.REWARD_ENGINEERING)
class CloseFeature(BaseFeature):

    source_data: DataFrame

    def __init__(
        self,
        dataset_id: DatasetIdentifier,
        indicator_service: ITechnicalIndicatorFacade,
        config: Optional[BaseParameterSetConfig] = None,
        postfix: str = ""
    ) -> None:
        super().__init__(dataset_id, indicator_service, config, postfix)
        self.feature_name = "close"

    def update(self, df: DataFrame) -> None:
        self.source_data = df

    def compute_latest(self) -> Optional[DataFrame]:
        if self.source_data is None or self.source_data.empty:
            return None
        latest_value = self.source_data["Close"].iloc[-1]
        # Return as DataFrame with single row
        import pandas as pd
        return pd.DataFrame([latest_value], columns=["close"], index=[self.source_data.index[-1]])

    def compute_all(self) -> Optional[DataFrame]:
        return self.source_data["Close"].to_frame()

    def _get_sub_features_names(self) -> list[str]:
        return ["value"]

    def _get_feature_type(self) -> FeatureTypeEnum:
        return get_feature_type_from_class(self.__class__)

    def _get_feature_name(self) -> str:
        return self._get_feature_type().value

    def _get_config_to_string(self) -> str:
        return "-"
