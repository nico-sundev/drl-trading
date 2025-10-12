
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List

import pandas as pd

from drl_trading_common.base import BaseFeature
from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo
from drl_trading_common.model.timeframe import Timeframe
from drl_trading_core.common.model.feature_coverage_summary import FeatureCoverageSummary
from drl_trading_core.common.model.feature_service_request_container import FeatureServiceRequestContainer


class IFeatureStoreFetchPort(ABC):
    @abstractmethod
    def get_online(
        self,
        feature_service_request: FeatureServiceRequestContainer,
    ) -> pd.DataFrame:
        """Fetch the most recent features for a given symbol and timeframe."""
        pass

    @abstractmethod
    def get_offline(
        self,
        feature_service_request: FeatureServiceRequestContainer,
        timestamps: pd.Series,
    ) -> pd.DataFrame:
        """
        Fetch historical features for multiple symbol-timeframe-timestamp rows.
        Assumes entity_df contains: 'symbol', 'timeframe', 'event_timestamp'.
        """
        pass

    @abstractmethod
    def get_feature_coverage_summary(
        self,
        features: List[BaseFeature],
        symbol: str,
        timeframe: Timeframe,
        start_time: datetime,
        end_time: datetime,
        feature_version_info: FeatureConfigVersionInfo
    ) -> Dict[str, FeatureCoverageSummary]:
        """
        Efficiently analyze feature coverage without fetching all data.

        This method:
        1. Maps BaseFeature instances to Feast field names internally
        2. Queries Feast for coverage metadata (counts, timestamps)
        3. Aggregates Feast fields back to business features
        4. Returns coverage keyed by feature.get_feature_name()

        This is 100-1000x faster than fetching all feature values because:
        - Only metadata is transferred (not actual feature values)
        - Uses Feast's aggregation capabilities
        - Minimizes memory usage

        Args:
            features: List of BaseFeature instances to analyze
            symbol: Trading symbol
            timeframe: Timeframe
            start_time: Start of period to analyze
            end_time: End of period to analyze
            feature_version_info: Feature configuration version information

        Returns:
            Dict mapping feature.get_feature_name() to FeatureCoverageSummary

        Example:
            >>> features = [RSIFeature(...), MACDFeature(...)]
            >>> coverage = port.get_feature_coverage_summary(
            ...     features, "EURUSD", Timeframe.HOUR_1, start, end
            ... )
            >>> coverage["rsi"].coverage_percentage
            95.5
            >>> coverage["macd"].is_fully_covered
            True
        """
        pass
