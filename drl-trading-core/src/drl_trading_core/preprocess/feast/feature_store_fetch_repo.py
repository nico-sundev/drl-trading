from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd
from drl_trading_common.models.timeframe import Timeframe
from feast import FeatureService, FeatureStore

# --- FEATURE FETCH REPOSITORY INTERFACE ---

class IFeatureFetchRepository(ABC):
    @abstractmethod
    def get_online(self, symbol: str, timeframe: Timeframe) -> Dict[str, Any]:
        """Fetch the most recent features for a given symbol and timeframe."""
        pass

    @abstractmethod
    def get_offline(self, entity_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fetch historical features for multiple symbol-timeframe-timestamp rows.
        Assumes entity_df contains: 'symbol', 'timeframe', 'event_timestamp'.
        """
        pass


class FeatureFetchRepository(IFeatureFetchRepository):
    def __init__(
        self,
        feature_store: FeatureStore,
        feature_service: FeatureService
    ):
        self._fs = feature_store
        self._feature_service = feature_service

    def get_online(self, symbol: str, timeframe: Timeframe) -> Dict[str, Any]:
        entity_rows = [{
            "symbol": symbol,
            "timeframe": timeframe.value
        }]
        feature_vector = self._fs.get_online_features(
            features=self._feature_service,
            entity_rows=entity_rows
        ).to_dict()
        return {k: v[0] for k, v in feature_vector.items()}

    def get_offline(self, entity_df: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"symbol", "timeframe", "event_timestamp"}
        if not required_cols.issubset(set(entity_df.columns)):
            raise ValueError(f"entity_df must contain columns: {required_cols}")

        return self._fs.get_historical_features(
            features=self._feature_service,
            entity_df=entity_df
        ).to_df()
