
from abc import ABC, abstractmethod

from drl_trading_common.model import FeatureConfigVersionInfo
import pandas as pd


class IFeatureStoreFetchPort(ABC):
    @abstractmethod
    def get_online(
        self,
        symbol: str,
        feature_version_info: FeatureConfigVersionInfo,
    ) -> pd.DataFrame:
        """Fetch the most recent features for a given symbol and timeframe."""
        pass

    @abstractmethod
    def get_offline(
        self,
        symbol: str,
        timestamps: pd.Series,
        feature_version_info: FeatureConfigVersionInfo,
    ) -> pd.DataFrame:
        """
        Fetch historical features for multiple symbol-timeframe-timestamp rows.
        Assumes entity_df contains: 'symbol', 'timeframe', 'event_timestamp'.
        """
        pass
