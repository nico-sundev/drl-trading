from abc import abstractmethod
from datetime import datetime
from typing import Optional

from .base_parameter_set_config import BaseParameterSetConfig
from .feature_metadata import FeatureMetadata
from ..port.feature_interface import IFeature
from drl_trading_common.decorator.feature_role_decorator import get_feature_role_from_class
from drl_trading_common.interface.computable import Computable
from drl_trading_common.interface.indicator.technical_indicator_facade_interface import ITechnicalIndicatorFacade
from drl_trading_common.adapter.model.dataset_identifier import DatasetIdentifier
from drl_trading_common.utils.utils import ensure_datetime_index
from pandas import DataFrame


class BaseFeature(Computable, IFeature):

    def __init__(
        self,
        dataset_id: DatasetIdentifier,
        indicator_service: ITechnicalIndicatorFacade,
        config: Optional[BaseParameterSetConfig] = None,
        postfix: str = "",
    ) -> None:
        self.config = config
        self.postfix = postfix
        self.indicator_service = indicator_service
        self.dataset_id = dataset_id
        # Store the last computed data for caught-up checking
        self.data: Optional[DataFrame] = None

    def _prepare_source_df(self, source: DataFrame, description: Optional[str] = None) -> DataFrame:
        """
        Prepares the source DataFrame for feature computation by ensuring it has a DatetimeIndex.

        Args:
            description: Optional description to use in logs instead of the default class name

        Returns:
            DataFrame: Source DataFrame with DatetimeIndex
        """
        feature_name = description or self._get_feature_name()
        return ensure_datetime_index(source, f"{feature_name} source data")

    @abstractmethod
    def _get_sub_features_names(self) -> list[str]:
        """Get the names of the sub-features.
        This method should be implemented by subclasses to return the names of the sub-features.

        Returns:
            list[str]: A list of sub-feature names.
        """
        ...

    @abstractmethod
    def _get_feature_name(self) -> str:
        ...

    @abstractmethod
    def _get_config_to_string(self) -> Optional[str]:
        ...

    def get_metadata(self) -> FeatureMetadata:
        """Get the metadata for the feature.

        Returns:
            FeatureMetadata: Metadata object containing feature information
        """
        return FeatureMetadata(
            config=self.config,
            dataset_id=self.dataset_id,
            feature_role=get_feature_role_from_class(self.__class__),
            feature_name=self._get_feature_name(),
            sub_feature_names=self._get_sub_features_names(),
            config_to_string=self._get_config_to_string()
        )

    def are_features_caught_up(self, reference_time: datetime) -> bool:
        """
        Check if the feature is caught up based on the last available record time.

        Compares the difference between the last available cached record and the reference time
        against the configured timeframe duration. If the difference is less than the timeframe
        duration, the feature is considered caught up.

        Args:
            reference_time: The current or target datetime to compare against

        Returns:
            True if the feature is caught up (time difference < timeframe duration), False otherwise
        """
        if self.data is None or self.data.empty:
            return False

        try:
            # Get the last timestamp from the cached data
            last_record_time = self.data.index[-1]

            # Convert to datetime if it's not already
            if hasattr(last_record_time, 'to_pydatetime'):
                last_record_time = last_record_time.to_pydatetime()
            elif isinstance(last_record_time, str):
                # Try to parse string timestamps
                import pandas as pd
                last_record_time = pd.to_datetime(last_record_time).to_pydatetime()
            elif not isinstance(last_record_time, datetime):
                # If it's not a datetime type, we can't determine catch-up status
                return False

            # Calculate time difference
            time_diff = reference_time - last_record_time

            # Get timeframe duration in seconds
            timeframe_duration_seconds = self.dataset_id.timeframe.to_seconds()

            # Feature is caught up if time difference is less than timeframe duration
            return time_diff.total_seconds() < timeframe_duration_seconds

        except Exception:
            # If any error occurs during time comparison, assume not caught up
            return False
