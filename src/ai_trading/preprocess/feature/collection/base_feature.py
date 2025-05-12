from abc import ABC, abstractmethod
from typing import Optional

from pandas import DataFrame

from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig
from ai_trading.data_set_utils.util import ensure_datetime_index
from ai_trading.preprocess.metrics.technical_metrics_service import (
    TechnicalMetricsServiceInterface,
)


class BaseFeature(ABC):

    def __init__(
        self,
        source: DataFrame,
        config: BaseParameterSetConfig,
        postfix: str = "",
        metrics_service: Optional[TechnicalMetricsServiceInterface] = None,
    ) -> None:
        self.df_source = source
        self.config = config
        self.postfix = postfix
        self.metrics_service = metrics_service

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
    def compute(self) -> DataFrame:
        pass

    @abstractmethod
    def get_sub_features_names(self) -> list[str]:
        """Get the names of the sub-features.
        This method should be implemented by subclasses to return the names of the sub-features.

        Returns:
            list[str]: A list of sub-feature names.
        """
        pass

    def get_feature_name(self) -> str:
        """Extract the feature name from the class name.

        For example, if the class name is 'RsiFeature', this method will return 'Rsi'.

        Returns:
            str: The base feature name without the 'Feature' suffix
        """
        class_name = self.__class__.__name__
        if class_name.endswith("Feature"):
            return class_name[:-7]  # Remove "Feature" suffix
        return class_name  # Return original name if it doesn't end with "Feature"
