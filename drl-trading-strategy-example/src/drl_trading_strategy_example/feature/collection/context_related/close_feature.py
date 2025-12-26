
from typing import Callable, Optional, override

from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_core.core.port.base_feature import BaseFeature
from drl_trading_core.core.service.feature.decorator.feature_role_decorator import feature_role
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_core.core.port.technical_indicator_service_port import (
    ITechnicalIndicatorServicePort,
)
from drl_trading_common.core.model.dataset_identifier import DatasetIdentifier
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
        indicator_service: ITechnicalIndicatorServicePort,
        config: Optional[BaseParameterSetConfig] = None,
        postfix: str = ""
    ) -> None:
        super().__init__(dataset_id, indicator_service, config, postfix)
        self.feature_name = "close"

    def update(self, df: DataFrame) -> None:
        self.source_data = df

    @override
    def compute_latest(self) -> Optional[DataFrame]:
        if self.source_data is None or self.source_data.empty:
            return None
        latest_value = self.source_data["Close"].iloc[-1]
        # Return as DataFrame with single row
        import pandas as pd
        feature_base_name = self.get_metadata().__str__()
        return pd.DataFrame([latest_value], columns=[feature_base_name], index=[self.source_data.index[-1]])

    @override
    def compute_all(self) -> Optional[DataFrame]:
        if self.source_data is None or self.source_data.empty:
            return None
        values = self.source_data["Close"]
        import pandas as pd
        feature_base_name = self.get_metadata().__str__()
        return pd.DataFrame([values], columns=[feature_base_name], index=[self.source_data.index])

    def _call_indicator_backend(self, method_call: Callable[[str], Optional[DataFrame]]) -> Optional[DataFrame]:
        return None

    def _get_sub_features_names(self) -> list[str]:
        return ["value"]

    def _get_feature_type(self) -> FeatureTypeEnum:
        return get_feature_type_from_class(self.__class__)

    def _get_feature_name(self) -> str:
        return self._get_feature_type().value

    def _get_config_to_string(self) -> str:
        return "-"
