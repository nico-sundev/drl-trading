
from abc import ABC, abstractmethod

import pandas as pd

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
