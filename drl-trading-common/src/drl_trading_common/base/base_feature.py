from abc import abstractmethod
from typing import Optional

from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_common.decorator.feature_role_decorator import get_feature_role_from_class
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.interface.computable import Computable
from drl_trading_common.interface.indicator.technical_indicator_facade_interface import ITechnicalIndicatorFacade
from drl_trading_common.model.dataset_identifier import DatasetIdentifier
from drl_trading_common.utils.utils import ensure_datetime_index
from pandas import DataFrame


class BaseFeature(Computable):

    def __init__(
        self,
        config: BaseParameterSetConfig,
        dataset_id: DatasetIdentifier,
        indicator_service: ITechnicalIndicatorFacade,
        postfix: str = "",
    ) -> None:
        self.config = config
        self.postfix = postfix
        self.indicator_service = indicator_service
        self.dataset_id = dataset_id

    def _prepare_source_df(self, source: DataFrame, description: Optional[str] = None) -> DataFrame:
        """
        Prepares the source DataFrame for feature computation by ensuring it has a DatetimeIndex.

        Args:
            description: Optional description to use in logs instead of the default class name

        Returns:
            DataFrame: Source DataFrame with DatetimeIndex
        """
        feature_name = description or self.get_feature_name()
        return ensure_datetime_index(source, f"{feature_name} source data")

    def get_config(self) -> BaseParameterSetConfig:
        """Get the configuration for the feature."""
        return self.config

    def get_dataset_id(self) -> DatasetIdentifier:
        """Get the dataset identifier for the feature."""
        return self.dataset_id

    def get_feature_role(self) -> FeatureRoleEnum:
        return get_feature_role_from_class(self.__class__)

    @abstractmethod
    def get_sub_features_names(self) -> list[str]:
        """Get the names of the sub-features.
        This method should be implemented by subclasses to return the names of the sub-features.

        Returns:
            list[str]: A list of sub-feature names.
        """
        pass

    @abstractmethod
    def get_feature_name(self) -> str:
        pass
