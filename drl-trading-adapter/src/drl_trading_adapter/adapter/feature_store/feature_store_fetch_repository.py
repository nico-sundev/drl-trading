import logging
from datetime import datetime
from typing import Dict, List

import pandas as pd
from injector import inject

from drl_trading_adapter.adapter.feature_store.provider import FeastProvider
from drl_trading_adapter.adapter.feature_store.provider.mapper.feature_field_mapper import FeatureFieldMapper
from drl_trading_adapter.adapter.feature_store.util.feature_store_utilities import (
    get_feature_service_name,
)
from drl_trading_common.base import BaseFeature
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo
from drl_trading_common.model.timeframe import Timeframe
from drl_trading_core.common.model.feature_coverage_summary import FeatureCoverageSummary
from drl_trading_core.common.model.feature_service_request_container import (
    FeatureServiceRequestContainer,
)
from drl_trading_core.core.port import IFeatureStoreFetchPort

logger = logging.getLogger(__name__)


@inject
class FeatureStoreFetchRepository(IFeatureStoreFetchPort):
    def __init__(self, feast_provider: FeastProvider):
        self._feast_provider = feast_provider
        self._fs = self._feast_provider.get_feature_store()
        self._field_mapper = FeatureFieldMapper()

    def get_online(
        self,
        feature_service_request: FeatureServiceRequestContainer,
    ) -> pd.DataFrame:
        entity_rows = [
            {
                "symbol": feature_service_request.symbol,
            }
        ]

        service_name = get_feature_service_name(request=feature_service_request)

        feature_service = self._feast_provider.get_feature_service(
            service_name=service_name
        )

        return self._fs.get_online_features(
            features=feature_service, entity_rows=entity_rows
        ).to_df()

    def get_offline(
        self,
        feature_service_request: FeatureServiceRequestContainer,
        timestamps: pd.Series,
    ) -> pd.DataFrame:
        service_name = get_feature_service_name(request=feature_service_request)

        feature_service = self._feast_provider.get_feature_service(
            service_name=service_name
        )

        symbol = feature_service_request.symbol

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
            with pd.option_context("future.no_silent_downcasting", True):
                result = self._fs.get_historical_features(
                    features=feature_service, entity_df=entity_df
                ).to_df()
                return result

        except Exception as e:
            logger.error(
                f"Unexpected error during historical features fetch for {symbol}: {e}"
            )
            # For unexpected errors, we should still raise them
            raise RuntimeError(
                f"Unexpected error during historical features fetch for {symbol}: {e}"
            ) from e

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

        Strategy:
        1. Map BaseFeature instances to Feast field names using FeatureFieldMapper
        2. Fetch features from Feast (unavoidable - Feast doesn't support metadata-only queries)
        3. Analyze coverage per feature by aggregating its Feast fields
        4. Return coverage keyed by feature.get_feature_name()

        Note: While we still fetch data from Feast (limitation of Feast API),
        this method provides the correct mapping between business features and Feast fields.
        Future optimization: Implement Feast metadata query support.
        """
        logger.info(
            f"Analyzing coverage for {len(features)} features: "
            f"{symbol} {timeframe.value} [{start_time} - {end_time}]"
        )

        try:
            # Generate timestamps for the period
            timestamps = pd.date_range(
                start=start_time,
                end=end_time,
                freq=timeframe.to_pandas_freq()
            )

            if len(timestamps) == 0:
                logger.warning(f"No timestamps in period [{start_time} - {end_time}]")
                return self._create_empty_coverage(features, 0)

            total_expected = len(timestamps)

            # Ensure timestamps are timezone-aware
            if timestamps.tz is None:
                timestamps = timestamps.tz_localize("UTC")

            # Build feature-to-fields mapping
            feature_to_fields = {}
            for feature in features:
                feature_name = feature.get_feature_name()
                feast_fields = self._field_mapper.create_fields(feature)
                feast_field_names = [field.name for field in feast_fields]
                feature_to_fields[feature_name] = feast_field_names
                logger.debug(
                    f"Feature '{feature_name}' maps to Feast fields: {feast_field_names}"
                )

            # Fetch features using get_offline method for consistency
            # Build feature service request with provided version info
            if not features:
                return {}

            feature_service_request = FeatureServiceRequestContainer(
                feature_service_role=FeatureRoleEnum.OBSERVATION_SPACE,
                symbol=symbol,
                feature_version_info=feature_version_info,
                timeframe=timeframe
            )

            # Reuse get_offline for consistency
            try:
                features_df = self.get_offline(
                    feature_service_request=feature_service_request,
                    timestamps=pd.Series(timestamps)
                )
            except Exception as fetch_error:
                logger.warning(
                    f"Could not fetch features for coverage analysis: {fetch_error}. "
                    "Returning empty coverage."
                )
                return self._create_empty_coverage(features, total_expected)

            # Analyze coverage for each feature
            coverage_summaries = {}

            for feature in features:
                feature_name = feature.get_feature_name()
                feast_field_names = feature_to_fields[feature_name]

                # Aggregate coverage across all Feast fields for this feature
                # A feature is considered "covered" if ANY of its fields has data
                combined_coverage = self._analyze_feature_fields_coverage(
                    features_df,
                    feast_field_names,
                    total_expected
                )

                coverage_summaries[feature_name] = FeatureCoverageSummary(
                    feature_name=feature_name,
                    total_expected_records=total_expected,
                    non_null_record_count=combined_coverage["non_null_count"],
                    null_record_count=combined_coverage["null_count"],
                    earliest_non_null_timestamp=combined_coverage["earliest_ts"],
                    latest_non_null_timestamp=combined_coverage["latest_ts"]
                )

                logger.debug(
                    f"Coverage for '{feature_name}': "
                    f"{coverage_summaries[feature_name].coverage_percentage:.1f}%"
                )

            logger.info(
                f"Coverage analysis complete: {len(coverage_summaries)} features analyzed"
            )

            return coverage_summaries

        except Exception as e:
            logger.error(f"Error analyzing feature coverage: {e}")
            return self._create_empty_coverage(features, 0)

    def _analyze_feature_fields_coverage(
        self,
        features_df: pd.DataFrame,
        feast_field_names: List[str],
        total_expected: int
    ) -> Dict:
        """
        Analyze coverage for a feature across its Feast fields.

        A timestamp is considered "covered" if ANY of the feature's fields has data.
        """
        if features_df.empty:
            return {
                "non_null_count": 0,
                "null_count": total_expected,
                "earliest_ts": None,
                "latest_ts": None
            }

        # Find which fields exist in the DataFrame
        existing_fields = [f for f in feast_field_names if f in features_df.columns]

        if not existing_fields:
            # None of the feature's fields exist in Feast
            return {
                "non_null_count": 0,
                "null_count": total_expected,
                "earliest_ts": None,
                "latest_ts": None
            }

        # Create a mask where ANY field is non-null for each timestamp
        combined_mask = pd.Series(False, index=features_df.index)
        for field_name in existing_fields:
            combined_mask |= features_df[field_name].notna()

        non_null_count = combined_mask.sum()
        null_count = total_expected - non_null_count

        # Get timestamps where feature has data
        if non_null_count > 0:
            covered_indices = features_df[combined_mask].index
            earliest_ts = covered_indices.min()
            latest_ts = covered_indices.max()

            # Convert to datetime if needed
            if hasattr(earliest_ts, 'to_pydatetime'):
                earliest_ts = earliest_ts.to_pydatetime()
                latest_ts = latest_ts.to_pydatetime()
        else:
            earliest_ts = None
            latest_ts = None

        return {
            "non_null_count": int(non_null_count),
            "null_count": int(null_count),
            "earliest_ts": earliest_ts,
            "latest_ts": latest_ts
        }

    def _create_empty_coverage(
        self,
        features: List[BaseFeature],
        total_expected: int
    ) -> Dict[str, FeatureCoverageSummary]:
        """Create empty coverage summaries for all features."""
        return {
            feature.get_feature_name(): FeatureCoverageSummary(
                feature_name=feature.get_feature_name(),
                total_expected_records=total_expected,
                non_null_record_count=0,
                null_record_count=total_expected,
                earliest_non_null_timestamp=None,
                latest_non_null_timestamp=None
            )
            for feature in features
        }
