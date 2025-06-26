from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from drl_trading_common.model.feature_config_version_info import (
    FeatureConfigVersionInfo,
)
from feast import FeatureService, FeatureStore
from injector import inject

from drl_trading_core.preprocess.feature_store.provider.feast_provider import (
    FeastProvider,
)


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
    def __init__(self, feature_store: FeatureStore, feast_provider: FeastProvider):
        self._fs = feature_store
        self._feast_provider = feast_provider
        self._feature_service: Optional[FeatureService] = None

    def get_online(
        self,
        symbol: str,
        feature_version_info: FeatureConfigVersionInfo,
    ) -> pd.DataFrame:
        entity_rows = [
            {
                "symbol": symbol,
            }
        ]

        if not self._feature_service:
            self._feature_service = self._feast_provider.create_feature_service(
                symbol=symbol, feature_version_info=feature_version_info
            )

        if self._feature_service is None:
            raise RuntimeError(
                "FeatureService is not initialized. Cannot fetch online features."
            )

        return self._fs.get_online_features(
            features=self._feature_service, entity_rows=entity_rows
        ).to_df()

    def get_offline(
        self,
        symbol: str,
        timestamps: pd.Series,
        feature_version_info: FeatureConfigVersionInfo,
    ) -> pd.DataFrame:

        if not self._feature_service:
            self._feature_service = self._feast_provider.create_feature_service(
                symbol=symbol, feature_version_info=feature_version_info
            )

        if self._feature_service is None:
            raise RuntimeError(
                "FeatureService is not initialized. Cannot fetch online features."
            )
        entity_df = pd.DataFrame()
        entity_df["event_timestamp"] = timestamps
        entity_df["symbol"] = symbol

        return self._fs.get_historical_features(
            features=self._feature_service, entity_df=entity_df
        ).to_df()
