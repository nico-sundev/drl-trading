
from abc import ABC, abstractmethod

import pandas as pd

from drl_trading_core.core.dto.feature_service_metadata import FeatureServiceMetadata


class IFeatureStoreFetchPort(ABC):
    @abstractmethod
    def get_online(
        self,
        feature_service_request: FeatureServiceMetadata,
    ) -> pd.DataFrame:
        """Fetch the most recent features for a given symbol and timeframe."""
        pass

    @abstractmethod
    def get_offline(
        self,
        feature_service_request: FeatureServiceMetadata,
        timestamps: pd.Series,
    ) -> pd.DataFrame:
        """
        Fetch historical features for multiple symbol-timeframe-timestamp rows.
        Assumes entity_df contains: 'symbol', 'timeframe', 'event_timestamp'.
        """
        pass
