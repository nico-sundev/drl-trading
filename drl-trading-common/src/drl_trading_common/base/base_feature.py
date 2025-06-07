from abc import ABC, abstractmethod
from typing import Optional

from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_common.interfaces.technical_indicator_service_interface import TechnicalIndicatorFacadeInterface
from drl_trading_common.utils.utils import ensure_datetime_index
from pandas import DataFrame


class BaseFeature(ABC):

    def __init__(
        self,
        source: DataFrame,
        config: BaseParameterSetConfig,
        indicator_service: TechnicalIndicatorFacadeInterface,
        postfix: str = "",
    ) -> None:
        self.df_source = source
        self.config = config
        self.postfix = postfix
        self.indicator_service = indicator_service

    def _prepare_source_df(self, description: Optional[str] = None) -> DataFrame:
        """
        Prepares the source DataFrame for feature computation by ensuring it has a DatetimeIndex.

        Args:
            description: Optional description to use in logs instead of the default class name

        Returns:
            DataFrame: Source DataFrame with DatetimeIndex
        """
        feature_name = description or self.get_feature_name()
        return ensure_datetime_index(self.df_source, f"{feature_name} source data")

    @abstractmethod
    def compute_all(self) -> Optional[DataFrame]:
        pass

    @abstractmethod
    def add(self, df: DataFrame) -> None:
        """Add new data to the feature. This method should be implemented by subclasses to handle new data."""
        pass

    @abstractmethod
    def compute_latest(self) -> Optional[DataFrame]:
        pass

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
