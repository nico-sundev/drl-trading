
from typing import Callable, Optional

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
from drl_trading_strategy_example.feature.config import RsiConfig
from drl_trading_strategy_example.mapper.mapper import TypeMapper
from pandas import DataFrame


@feature_type(FeatureTypeEnum.RSI)
@feature_role(FeatureRoleEnum.OBSERVATION_SPACE)
class RsiFeature(BaseFeature):

    def __init__(
        self,
        dataset_id: DatasetIdentifier,
        indicator_service: ITechnicalIndicatorFacade,
        config: Optional[BaseParameterSetConfig] = None,
        postfix: str = ""
    ) -> None:
        super().__init__(dataset_id, indicator_service, config, postfix)
        if config is None:
            raise ValueError("RsiFeature requires a configuration with length parameter")
        if not isinstance(config, RsiConfig):
            raise TypeError(f"RsiFeature requires config to be of type RsiConfig, got {type(config).__name__}")
        self.config: RsiConfig = config
        self.feature_name = f"rsi_{self.config.length}{self.postfix}"
        self.indicator_service.register_instance(self.feature_name, self._get_indicator_type(), config=self.config)

    def update(self, df: DataFrame) -> None:
        index_corrected_dataframe = self._prepare_source_df(df)
        self.indicator_service.add(self.feature_name, index_corrected_dataframe)

    def compute_latest(self) -> Optional[DataFrame]:
        """
        Get latest RSI value with timestamp.

        Returns:
            DataFrame with DatetimeIndex and latest RSI value
        """
        return self._call_indicator_backend(self.indicator_service.get_latest)

    def compute_all(self) -> Optional[DataFrame]:
        """
        Get all RSI values with timestamps.

        Returns:
            DataFrame with DatetimeIndex containing all RSI values.
            The index preserves the original timestamps from the OHLCV data,
            enabling proper time-based partitioning in the feature store.
        """
        return self._call_indicator_backend(self.indicator_service.get_all)

    def _call_indicator_backend(self, method_call: Callable[[str], Optional[DataFrame]]) -> Optional[DataFrame]:
        """
        Generic method to call indicator backend methods in a thread-safe manner.

        Args:
            method_call: A callable that takes the feature name and returns a DataFrame.

        Returns:
            DataFrame with DatetimeIndex containing the result of the method call.
        """
        result = method_call(self.feature_name)
        if result is None:
            return None

        # Rename column to match feature name (indicator returns "rsi", we want "rsi_14" etc.)
        result.columns = [f"rsi_{self.config.length}{self.postfix}"]
        return result

    def _get_sub_features_names(self) -> list[str]:
        return ["value"]

    def _get_feature_type(self) -> FeatureTypeEnum:
        return get_feature_type_from_class(self.__class__)

    def _get_indicator_type(self) -> str:
        """Get the indicator type as a string for decoupling from enums."""
        enum_type = TypeMapper.map_feature_to_indicator(self._get_feature_type())
        return enum_type.value

    def _get_feature_name(self) -> str:
        return self._get_feature_type().value

    def _get_config_to_string(self) -> str:
        return f"{self.config.length}"
