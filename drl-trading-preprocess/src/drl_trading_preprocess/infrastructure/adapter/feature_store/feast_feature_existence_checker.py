"""
Feast-based implementation of feature existence checking.

Provides concrete implementation for checking feature existence
in Feast feature store to support incremental processing scenarios.
"""
import logging
from datetime import datetime

from injector import inject
from pandas import DataFrame
import pandas as pd

from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo
from drl_trading_common.model.timeframe import Timeframe
from drl_trading_core.common.model.feature_service_request_container import FeatureServiceRequestContainer
from drl_trading_core.core.port.feature_store_fetch_port import IFeatureStoreFetchPort
from drl_trading_core.core.service.feature_manager import FeatureManager

logger = logging.getLogger(__name__)


@inject
class FeastFeatureExistenceChecker:
    """
    Feast-based implementation for checking feature existence.

    Provides comprehensive feature existence checking including:
    - Basic existence verification
    - Temporal coverage analysis
    - Schema version compatibility
    - Gap detection for incremental processing
    """

    def __init__(self, feature_store_fetch_port: IFeatureStoreFetchPort, feature_manager: FeatureManager) -> None:
        """Initialize the Feast feature existence checker."""
        self.feature_store_fetch_port = feature_store_fetch_port
        self.feature_manager = feature_manager

    def fetch_from_feature_store(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        feature_name: str,
        feature_config_version_info: FeatureConfigVersionInfo,
        timeframe: Timeframe,
    ) -> DataFrame:
        """
        Check which features exist for the given symbol and time range.

        Args:
            symbol: Trading symbol to check
            timeframe: Timeframe for the features
            feature_names: List of feature names to check
            start_time: Start of time range to check
            end_time: End of time range to check
            feature_config_version_info: Feature configuration version info

        Returns:
            DataFrame with detailed existence information
        """

        request = FeatureServiceRequestContainer.create(
            symbol=symbol,
            feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature_config_version=feature_config_version_info,
            timeframe=timeframe,
        )
        # Generate a Series of timestamps from start_time to end_time (inclusive)
        timestamps = DataFrame({"timestamp": pd.date_range(start=start_time, end=end_time, freq=timeframe.to_pandas_freq())})["timestamp"]

        # Fetch features from the feature store
        feature_df = self.feature_store_fetch_port.get_offline(request, timestamps)

        # TODO: Integrate timestamps with feature_df for existence checking

        return feature_df
