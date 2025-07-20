import logging
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from drl_trading_common.model.feature_config_version_info import (
    FeatureConfigVersionInfo,
)
from feast import FeatureService
from injector import inject

from drl_trading_core.preprocess.feature_store.provider.feast_provider import (
    FeastProvider,
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
    def __init__(self, feast_provider: FeastProvider):
        self._feast_provider = feast_provider
        self._fs = self._feast_provider.get_feature_store()
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

        # Validate and clean timestamps to prevent Dask/Feast issues
        if timestamps.isnull().any():
            logger.warning(f"Found null values in timestamps for {symbol}, dropping nulls")
            timestamps = timestamps.dropna()

        if timestamps.empty:
            logger.warning(f"No valid timestamps to fetch for {symbol}")
            return pd.DataFrame()

        # Ensure timestamps are timezone-aware to prevent Feast timezone issues
        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize("UTC")

        entity_df = pd.DataFrame()
        entity_df["event_timestamp"] = timestamps
        entity_df["symbol"] = symbol

        try:
            result = self._fs.get_historical_features(
                features=self._feature_service, entity_df=entity_df
            ).to_df()

            return result

        except (NotImplementedError, TypeError) as e:
            if "divisions calculation failed" in str(e) or "nulls" in str(e):
                logger.warning(f"Dask divisions calculation failed for {symbol}: {e}")
                logger.info("This is a known compatibility issue between Dask and Feast versions")
                logger.info("Consider using a different offline store backend or upgrading library versions")
                # Return empty DataFrame as fallback
                return pd.DataFrame()
            else:
                raise RuntimeError(f"Historical features fetch failed for {symbol}: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during historical features fetch for {symbol}: {e}")
            # For unexpected errors, we should still raise them
            raise RuntimeError(f"Unexpected error during historical features fetch for {symbol}: {e}") from e
