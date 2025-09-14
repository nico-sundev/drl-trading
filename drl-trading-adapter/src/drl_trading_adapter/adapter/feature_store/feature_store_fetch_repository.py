import datetime
import logging

import pandas as pd
from feast import FeatureService
from injector import inject

from drl_trading_adapter.adapter.feature_store.util.feature_store_utilities import (
    get_feature_service_name,
)
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

        # entity_df = pd.DataFrame()
        # entity_df["event_timestamp"] = timestamps
        # entity_df["symbol"] = symbol
        entity_df = pd.DataFrame.from_dict(
            {
                "symbol": ["EURUSD", "EURUSD"],
                "event_timestamp": [
                    datetime.datetime(2021, 4, 12, 10, 59, 42),
                    datetime.datetime(2021, 4, 12, 8, 12, 10),
                ],
            }
        )

        # Validate entity DataFrame before passing to Feast
        self._validate_entity_dataframe(entity_df, symbol)

        try:
            logger.debug(
                f"Fetching historical features for {symbol} with {len(entity_df)} timestamps"
            )
            logger.debug(f"Entity DataFrame schema: {entity_df.dtypes.to_dict()}")
            logger.debug(f"Feature service: {feature_service.name}")

            result = self._fs.get_historical_features(
                features=feature_service, entity_df=entity_df
            ).to_df()

            logger.info(
                f"Successfully fetched {len(result)} historical feature records for {symbol}"
            )
            return result

        except Exception as e:
            # Provide detailed diagnostic information for debugging
            logger.error(f"Historical features fetch failed for {symbol}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Entity DataFrame shape: {entity_df.shape}")
            logger.error(f"Entity DataFrame columns: {list(entity_df.columns)}")
            logger.error(f"Entity DataFrame dtypes: {entity_df.dtypes.to_dict()}")
            logger.error(f"Feature service name: {feature_service.name}")
            logger.error(
                f"Feature service features: {[f.name for f in feature_service._features] if hasattr(feature_service, '_features') else 'N/A'}"
            )

            # Check for specific known issues with detailed context
            error_msg = str(e).lower()
            if "divisions calculation failed" in error_msg:
                self._diagnose_divisions_error(entity_df, symbol, e)
            elif "file not found" in error_msg or "no such file" in error_msg:
                self._diagnose_file_source_error(entity_df, feature_service, symbol, e)
            elif "schema" in error_msg or "column" in error_msg:
                self._diagnose_schema_mismatch_error(
                    entity_df, feature_service, symbol, e
                )
            else:
                logger.error("Unrecognized error pattern. This requires investigation.")

            # Always re-raise - don't silently fail
            raise RuntimeError(
                f"Failed to fetch historical features for {symbol}. "
                f"Check logs above for detailed diagnostics. Original error: {e}"
            ) from e

    def _validate_entity_dataframe(self, entity_df: pd.DataFrame, symbol: str) -> None:
        """Validate entity DataFrame before passing to Feast."""
        if entity_df.empty:
            raise ValueError(f"Entity DataFrame is empty for symbol {symbol}")

        if "event_timestamp" not in entity_df.columns:
            raise ValueError(
                f"Entity DataFrame missing 'event_timestamp' column for symbol {symbol}"
            )

        if "symbol" not in entity_df.columns:
            raise ValueError(
                f"Entity DataFrame missing 'symbol' column for symbol {symbol}"
            )

        # Check for null timestamps
        null_timestamps = entity_df["event_timestamp"].isnull().sum()
        if null_timestamps > 0:
            logger.warning(
                f"Found {null_timestamps} null timestamps in entity DataFrame for {symbol}"
            )
            # Don't fail here, but log it for diagnostics

    def _diagnose_divisions_error(
        self, entity_df: pd.DataFrame, symbol: str, error: Exception
    ) -> None:
        """Diagnose Dask divisions calculation errors."""
        logger.error("DIAGNOSIS: Dask divisions calculation failed")
        logger.error(
            "This usually indicates an issue with the timestamp index or file source"
        )
        logger.error("Timestamp column info:")
        logger.error(f"  - Null count: {entity_df['event_timestamp'].isnull().sum()}")
        logger.error(f"  - Dtype: {entity_df['event_timestamp'].dtype}")
        logger.error(f"  - Min timestamp: {entity_df['event_timestamp'].min()}")
        logger.error(f"  - Max timestamp: {entity_df['event_timestamp'].max()}")

    def _diagnose_file_source_error(
        self,
        entity_df: pd.DataFrame,
        feature_service: FeatureService,
        symbol: str,
        error: Exception,
    ) -> None:
        """Diagnose file source related errors."""
        logger.error("DIAGNOSIS: File source error detected")
        logger.error("This indicates that Feast cannot find the offline data files")
        logger.error("Possible causes:")
        logger.error("  - Feature view file sources point to non-existent files")
        logger.error("  - Offline repository failed to create/store parquet files")
        logger.error("  - File permissions or path issues")
        for feature_view in feature_service._features:
            if hasattr(feature_view, "source") and hasattr(feature_view.source, "path"):
                logger.error(
                    f"  - Feature view '{feature_view.name}' source path: {feature_view.source.path}"
                )

    def _diagnose_schema_mismatch_error(
        self,
        entity_df: pd.DataFrame,
        feature_service: FeatureService,
        symbol: str,
        error: Exception,
    ) -> None:
        """Diagnose schema mismatch errors."""
        logger.error("DIAGNOSIS: Schema mismatch detected")
        logger.error(
            "This indicates a mismatch between entity DataFrame and feature view schemas"
        )
        logger.error(f"Entity DataFrame columns: {list(entity_df.columns)}")
        logger.error(f"Entity DataFrame dtypes: {entity_df.dtypes.to_dict()}")
        logger.error("Feature service feature views:")
        for feature_view in feature_service._features:
            logger.error(
                f"  - Feature view '{feature_view.name}' schema: {[f.name for f in feature_view.schema] if hasattr(feature_view, 'schema') else 'N/A'}"
            )
