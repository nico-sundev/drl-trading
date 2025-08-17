import logging
from abc import ABC, abstractmethod

import pandas as pd
from drl_trading_common.model.feature_config_version_info import (
    FeatureConfigVersionInfo,
)
from injector import inject

from drl_trading_core.preprocess.feature_store.port.feature_store_operation_ports import (
    IOnlineFeatureReader,
    IHistoricalFeatureReader,
)

logger = logging.getLogger(__name__)


class IFeatureStoreFetchRepository(ABC):
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


@inject
class FeatureStoreFetchRepository(IFeatureStoreFetchRepository):
    def __init__(self, online_reader: IOnlineFeatureReader, historical_reader: IHistoricalFeatureReader):
        self._online_reader = online_reader
        self._historical_reader = historical_reader

    def get_online(
        self,
        symbol: str,
        feature_version_info: FeatureConfigVersionInfo,
    ) -> pd.DataFrame:
        return self._online_reader.get_online(symbol=symbol, feature_version_info=feature_version_info)

    def get_offline(
        self,
        symbol: str,
        timestamps: pd.Series,
        feature_version_info: FeatureConfigVersionInfo,
    ) -> pd.DataFrame:
        return self._historical_reader.get_offline(symbol=symbol, timestamps=timestamps, feature_version_info=feature_version_info)
