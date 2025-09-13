import logging

import pandas as pd
from injector import inject

from drl_trading_adapter.adapter.feature_store.util.feature_store_utilities import get_feature_service_name
from drl_trading_adapter.adapter.feature_store.provider import FeastProvider
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.model.feature_config_version_info import (
    FeatureConfigVersionInfo,
)
from drl_trading_core.core.port import IFeatureStoreFetchPort

logger = logging.getLogger(__name__)


@inject
class FeatureStoreFetchRepository(IFeatureStoreFetchPort):
    def __init__(self, feast_provider: FeastProvider):
        self._feast_provider = feast_provider
        self._fs = self._feast_provider.get_feature_store()

    def get_online(
        self,
        symbol: str,
        feature_version_info: FeatureConfigVersionInfo,
        feature_service_role: FeatureRoleEnum,
    ) -> pd.DataFrame:
        entity_rows = [
            {
                "symbol": symbol,
            }
        ]

        service_name = get_feature_service_name(
            feature_service_role=feature_service_role,
            symbol=symbol,
            feature_version_info=feature_version_info,
        )

        feature_service = self._feast_provider.get_feature_service(
            service_name=service_name
        )

        return self._fs.get_online_features(
            features=feature_service, entity_rows=entity_rows
        ).to_df()

    def get_offline(
        self,
        symbol: str,
        timestamps: pd.Series,
        feature_version_info: FeatureConfigVersionInfo,
        feature_service_role: FeatureRoleEnum,
    ) -> pd.DataFrame:

        service_name = get_feature_service_name(
            feature_service_role=feature_service_role,
            symbol=symbol,
            feature_version_info=feature_version_info,
        )

        feature_service = self._feast_provider.get_feature_service(
            service_name=service_name
        )

        # Validate and clean timestamps to prevent Dask/Feast issues
        if timestamps.isnull().any():
            logger.warning(
                f"Found null values in timestamps for {symbol}, dropping nulls"
            )
            timestamps = timestamps.dropna()

        if timestamps.empty:
            logger.warning(f"No valid timestamps to fetch for {symbol}")
            return pd.DataFrame()

        # Ensure timestamps are timezone-aware to prevent Feast timezone issues
        # Handle both Series and DatetimeIndex cases
        if hasattr(timestamps, "dt"):
            # timestamps is a Series
            if timestamps.dt.tz is None:
                timestamps = timestamps.dt.tz_localize("UTC")
        else:
            # timestamps is a DatetimeIndex
            if timestamps.tz is None:
                timestamps = timestamps.tz_localize("UTC")

        entity_df = pd.DataFrame()
        entity_df["event_timestamp"] = timestamps
        entity_df["symbol"] = symbol

        try:
            result = self._fs.get_historical_features(
                features=feature_service, entity_df=entity_df
            ).to_df()

            return result

        except (NotImplementedError, TypeError) as e:
            if "divisions calculation failed" in str(e) or "nulls" in str(e):
                logger.warning(f"Dask divisions calculation failed for {symbol}: {e}")
                logger.info(
                    "This is a known compatibility issue between Dask and Feast versions"
                )
                logger.info(
                    "Consider using a different offline store backend or upgrading library versions"
                )
                # Return empty DataFrame as fallback
                return pd.DataFrame()
            else:
                raise RuntimeError(
                    f"Historical features fetch failed for {symbol}: {e}"
                ) from e
        except Exception as e:
            logger.error(
                f"Unexpected error during historical features fetch for {symbol}: {e}"
            )
            # For unexpected errors, we should still raise them
            raise RuntimeError(
                f"Unexpected error during historical features fetch for {symbol}: {e}"
            ) from e
