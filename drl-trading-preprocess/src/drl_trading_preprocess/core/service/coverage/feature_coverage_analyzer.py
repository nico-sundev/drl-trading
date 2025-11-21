"""
Feature coverage analyzer for determining what features need computation.

This service analyzes feature coverage by:
1. Checking OHLCV data availability in TimescaleDB (constraining time period)
2. Batch fetching existing features from Feast
3. Analyzing coverage gaps and determining computation needs
4. Supporting both training and inference scenarios
"""
import logging
from datetime import datetime
from typing import Dict, List

import pandas as pd
from injector import inject
from pandas import DataFrame

from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo
from drl_trading_common.model.timeframe import Timeframe
from drl_trading_core.common.model.feature_service_request_container import FeatureServiceRequestContainer
from drl_trading_core.core.port.feature_store_fetch_port import IFeatureStoreFetchPort
from drl_trading_core.core.port.market_data_reader_port import MarketDataReaderPort
from drl_trading_preprocess.core.model.coverage.feature_coverage_analysis import (
    FeatureCoverageAnalysis,
    FeatureCoverageInfo,
    OhlcvAvailability,
)
from drl_trading_preprocess.core.service.coverage.feature_coverage_evaluator import (
    FeatureCoverageEvaluator,
)

logger = logging.getLogger(__name__)


@inject
class FeatureCoverageAnalyzer:
    """
    Analyzes feature coverage to determine computation requirements.

    Key responsibilities:
    - Check OHLCV data availability to constrain time periods
    - Batch fetch features from Feast for coverage analysis
    - Identify missing, partial, and fully covered features
    - Support both training (historical) and inference (real-time) scenarios
    - Provide warmup period calculations
    """

    def __init__(
        self,
        feature_store_fetch_port: IFeatureStoreFetchPort,
        market_data_reader: MarketDataReaderPort,
        feature_coverage_evaluator: FeatureCoverageEvaluator,
    ) -> None:
        """
        Initialize the feature coverage analyzer.

        Args:
            feature_store_fetch_port: Port for fetching features from Feast
            market_data_reader: Port for checking OHLCV data availability
            feature_coverage_evaluator: Evaluator for coverage analysis operations
        """
        self.feature_store_fetch_port = feature_store_fetch_port
        self.market_data_reader = market_data_reader
        self.feature_coverage_evaluator = feature_coverage_evaluator
        logger.info("FeatureCoverageAnalyzer initialized")

    def analyze_feature_coverage(
        self,
        symbol: str,
        timeframe: Timeframe,
        base_timeframe: Timeframe,
        feature_names: List[str],
        requested_start_time: datetime,
        requested_end_time: datetime,
        feature_config_version_info: FeatureConfigVersionInfo,
    ) -> FeatureCoverageAnalysis:
        """
        Analyze feature coverage for the given parameters.

        Process:
        1. Check OHLCV data availability to constrain the time period
        2. Batch fetch all requested features from Feast
        3. Analyze coverage for each feature
        4. Return comprehensive analysis

        Args:
            symbol: Trading symbol
            timeframe: Target timeframe
            base_timeframe: Base timeframe to fall back to if target doesn't exist
            feature_names: List of feature names to analyze
            requested_start_time: Requested start of time period
            requested_end_time: Requested end of time period
            feature_config_version_info: Feature configuration version

        Returns:
            FeatureCoverageAnalysis: Complete coverage analysis
        """
        logger.info(
            f"Analyzing feature coverage for {symbol} {timeframe.value}: "
            f"{len(feature_names)} features, period [{requested_start_time} - {requested_end_time}]"
        )

        # Step 1: Check OHLCV data availability in target timeframe
        ohlcv_availability = self._check_ohlcv_availability(
            symbol, timeframe, requested_start_time, requested_end_time
        )

        requires_resampling = False

        # Step 1b: Fallback to base timeframe if target doesn't exist (cold start scenario)
        if not ohlcv_availability.available:
            logger.info(
                f"No data in target timeframe {timeframe.value}, "
                f"checking base timeframe {base_timeframe.value} for cold start"
            )
            ohlcv_availability = self._check_ohlcv_availability(
                symbol, base_timeframe, requested_start_time, requested_end_time
            )
            if ohlcv_availability.available:
                requires_resampling = True
                logger.info(
                    f"Found {ohlcv_availability.record_count} records in base timeframe {base_timeframe.value}. "
                    f"Target timeframe {timeframe.value} will be resampled from base."
                )

        # Step 2: Determine adjusted time period based on OHLCV constraints
        adjusted_start_time = requested_start_time
        adjusted_end_time = requested_end_time

        # If no OHLCV data is available, return analysis indicating no data immediately
        if not ohlcv_availability.available:
            logger.error(
                f"No OHLCV data available for {symbol} {timeframe.value} "
                f"in period [{requested_start_time} - {requested_end_time}]"
            )
            # Return analysis indicating no data available
            return self._create_no_data_analysis(
                symbol, timeframe, feature_names,
                requested_start_time, requested_end_time,
                adjusted_start_time, adjusted_end_time
            )

        # Constrain to actual OHLCV data bounds
        if ohlcv_availability.earliest_timestamp:
            adjusted_start_time = max(
                requested_start_time,
                ohlcv_availability.earliest_timestamp
            )
        if ohlcv_availability.latest_timestamp:
            adjusted_end_time = min(
                requested_end_time,
                ohlcv_availability.latest_timestamp
            )

        logger.info(
            f"OHLCV availability for {symbol} {timeframe.value}: "
            f"{ohlcv_availability.record_count} records "
            f"[{ohlcv_availability.earliest_timestamp} - {ohlcv_availability.latest_timestamp}]"
        )

        if (adjusted_start_time != requested_start_time or
            adjusted_end_time != requested_end_time):
            logger.warning(
                f"Requested period adjusted by OHLCV availability: "
                f"[{requested_start_time} - {requested_end_time}] to "
                f"[{adjusted_start_time} - {adjusted_end_time}]"
            )


        # Step 3: Batch fetch existing features from Feast
        existing_features_df = self._batch_fetch_features(
            symbol=symbol,
            timeframe=timeframe,
            start_time=adjusted_start_time,
            end_time=adjusted_end_time,
            feature_config_version_info=feature_config_version_info
        )

        # Step 4: Analyze coverage for each feature
        feature_coverage = self._analyze_individual_features(
            feature_names=feature_names,
            existing_features_df=existing_features_df,
            adjusted_start_time=adjusted_start_time,
            adjusted_end_time=adjusted_end_time,
            timeframe=timeframe
        )

        # Step 5: Return comprehensive analysis
        analysis = FeatureCoverageAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            requested_start_time=requested_start_time,
            requested_end_time=requested_end_time,
            ohlcv_available=ohlcv_availability.available,
            ohlcv_earliest_timestamp=ohlcv_availability.earliest_timestamp,
            ohlcv_latest_timestamp=ohlcv_availability.latest_timestamp,
            ohlcv_record_count=ohlcv_availability.record_count,
            requires_resampling=requires_resampling,
            adjusted_start_time=adjusted_start_time,
            adjusted_end_time=adjusted_end_time,
            feature_coverage=feature_coverage,
            existing_features_df=existing_features_df
        )

        logger.info(self.feature_coverage_evaluator.get_summary_message(analysis))

        return analysis

    def _check_ohlcv_availability(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_time: datetime,
        end_time: datetime
    ) -> OhlcvAvailability:
        """
        Check OHLCV data availability in TimescaleDB.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_time: Start of period
            end_time: End of period

        Returns:
            OhlcvAvailability: Availability information
        """
        try:
            # Use database-side aggregation for efficient availability checking
            availability = self.market_data_reader.get_data_availability(
                symbol=symbol,
                timeframe=timeframe
            )

            if availability.record_count == 0:
                logger.warning(
                    f"No OHLCV data found for {symbol} {timeframe.value}"
                )
                return OhlcvAvailability(
                    available=False,
                    record_count=0,
                    earliest_timestamp=None,
                    latest_timestamp=None
                )

            # Check if requested period overlaps with available data
            has_overlap = (
                availability.earliest_timestamp is not None and
                availability.latest_timestamp is not None and
                availability.earliest_timestamp <= end_time and
                availability.latest_timestamp >= start_time
            )

            if not has_overlap:
                logger.warning(
                    f"Requested period [{start_time} - {end_time}] does not overlap "
                    f"with available OHLCV data [{availability.earliest_timestamp} - {availability.latest_timestamp}]"
                )
                return OhlcvAvailability(
                    available=False,
                    record_count=0,
                    earliest_timestamp=availability.earliest_timestamp,
                    latest_timestamp=availability.latest_timestamp
                )

            return OhlcvAvailability(
                available=True,
                record_count=availability.record_count,
                earliest_timestamp=availability.earliest_timestamp,
                latest_timestamp=availability.latest_timestamp
            )

        except Exception as e:
            logger.error(f"Error checking OHLCV availability: {e}")
            return OhlcvAvailability(
                available=False,
                record_count=0,
                earliest_timestamp=None,
                latest_timestamp=None
            )

    def _batch_fetch_features(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_time: datetime,
        end_time: datetime,
        feature_config_version_info: FeatureConfigVersionInfo,
    ) -> DataFrame:
        """
        Batch fetch all features from Feast for the given period.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_time: Start of period
            end_time: End of period
            feature_config_version_info: Feature configuration version

        Returns:
            DataFrame with fetched features (empty if none exist)
        """
        try:
            # Create feature service request
            request = FeatureServiceRequestContainer.create(
                symbol=symbol,
                feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
                feature_config_version=feature_config_version_info,
                timeframe=timeframe,
            )

            # Generate timestamps for the period
            timestamps = pd.date_range(
                start=start_time,
                end=end_time,
                freq=timeframe.to_pandas_freq()
            )

            if len(timestamps) == 0:
                logger.warning(
                    f"No timestamps generated for period [{start_time} - {end_time}] "
                    f"with frequency {timeframe.to_pandas_freq()}"
                )
                return DataFrame()

            logger.info(
                f"Fetching features for {len(timestamps)} timestamps "
                f"from {timestamps[0]} to {timestamps[-1]}"
            )

            # Fetch features from Feast
            features_df = self.feature_store_fetch_port.get_offline(
                request, pd.Series(timestamps)
            )

            if features_df.empty:
                logger.info(f"No existing features found in Feast for {symbol} {timeframe.value}")
            else:
                logger.info(
                    f"Fetched {len(features_df)} records with {len(features_df.columns)} columns "
                    f"from Feast"
                )

            return features_df

        except Exception as e:
            logger.error(f"Error fetching features from Feast: {e}")
            return DataFrame()

    def _analyze_individual_features(
        self,
        feature_names: List[str],
        existing_features_df: DataFrame,
        adjusted_start_time: datetime,
        adjusted_end_time: datetime,
        timeframe: Timeframe
    ) -> Dict[str, FeatureCoverageInfo]:
        """
        Analyze coverage for each individual feature.

        Args:
            feature_names: List of feature names to analyze
            existing_features_df: DataFrame with existing features
            adjusted_start_time: Start of analysis period
            adjusted_end_time: End of analysis period
            timeframe: Timeframe

        Returns:
            Dict mapping feature names to coverage info
        """
        feature_coverage = {}

        # Calculate expected number of records for the period
        expected_timestamps = pd.date_range(
            start=adjusted_start_time,
            end=adjusted_end_time,
            freq=timeframe.to_pandas_freq()
        )
        expected_count = len(expected_timestamps)

        for feature_name in feature_names:
            if existing_features_df.empty or feature_name not in existing_features_df.columns:
                # Feature doesn't exist at all
                feature_coverage[feature_name] = FeatureCoverageInfo(
                    feature_name=feature_name,
                    is_fully_covered=False,
                    earliest_timestamp=None,
                    latest_timestamp=None,
                    record_count=0,
                    coverage_percentage=0.0,
                    missing_periods=[(adjusted_start_time, adjusted_end_time)]
                )
            else:
                # Analyze existing feature data
                feature_series = existing_features_df[feature_name]
                non_null_series = feature_series.dropna()
                record_count = len(non_null_series)

                if record_count == 0:
                    # Feature exists but has no valid data
                    feature_coverage[feature_name] = FeatureCoverageInfo(
                        feature_name=feature_name,
                        is_fully_covered=False,
                        earliest_timestamp=None,
                        latest_timestamp=None,
                        record_count=0,
                        coverage_percentage=0.0,
                        missing_periods=[(adjusted_start_time, adjusted_end_time)]
                    )
                else:
                    # Feature has data - analyze coverage
                    earliest_ts = non_null_series.index.min()
                    latest_ts = non_null_series.index.max()
                    coverage_pct = (record_count / expected_count * 100.0) if expected_count > 0 else 0.0
                    is_fully_covered = coverage_pct >= 99.0  # Allow small tolerance

                    # Identify missing periods (gaps)
                    missing_periods = self._identify_missing_periods(
                        feature_series,
                        expected_timestamps
                    )

                    feature_coverage[feature_name] = FeatureCoverageInfo(
                        feature_name=feature_name,
                        is_fully_covered=is_fully_covered,
                        earliest_timestamp=earliest_ts,
                        latest_timestamp=latest_ts,
                        record_count=record_count,
                        coverage_percentage=coverage_pct,
                        missing_periods=missing_periods
                    )

        return feature_coverage

    def _identify_missing_periods(
        self,
        feature_series: pd.Series,
        expected_timestamps: pd.DatetimeIndex
    ) -> List[tuple[datetime, datetime]]:
        """
        Identify gaps/missing periods in feature data.

        Args:
            feature_series: Series with feature data
            expected_timestamps: Expected timestamps for full coverage

        Returns:
            List of (start, end) tuples representing missing periods
        """
        # Get timestamps with null values
        null_mask = feature_series.isnull()
        null_timestamps = feature_series[null_mask].index

        if len(null_timestamps) == 0:
            return []

        # Group consecutive null timestamps into periods
        missing_periods = []
        current_start = None
        previous_ts = None

        for ts in sorted(null_timestamps):
            if current_start is None:
                current_start = ts
            elif previous_ts is not None:
                # Check if there's a gap between previous and current
                # (non-consecutive timestamps)
                time_diff = (ts - previous_ts).total_seconds()

                # Calculate expected difference
                if expected_timestamps.freq is not None and hasattr(expected_timestamps.freq, 'nanos'):
                    expected_diff = expected_timestamps.freq.nanos / 1e9  # Convert to seconds
                elif len(expected_timestamps) > 1:
                    # Fallback: calculate from first two timestamps
                    expected_diff = (expected_timestamps[1] - expected_timestamps[0]).total_seconds()
                else:
                    # Default to a reasonable gap threshold (1 hour)
                    expected_diff = 3600.0

                if time_diff > expected_diff * 1.5:  # Allow some tolerance
                    # Gap found - close current period and start new one
                    missing_periods.append((current_start, previous_ts))
                    current_start = ts

            previous_ts = ts

        # Close the final period
        if current_start is not None and previous_ts is not None:
            missing_periods.append((current_start, previous_ts))

        return missing_periods

    def _create_no_data_analysis(
        self,
        symbol: str,
        timeframe: Timeframe,
        feature_names: List[str],
        requested_start_time: datetime,
        requested_end_time: datetime,
        adjusted_start_time: datetime,
        adjusted_end_time: datetime
    ) -> FeatureCoverageAnalysis:
        """
        Create analysis result when no OHLCV data is available.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            feature_names: List of feature names
            requested_start_time: Requested start time
            requested_end_time: Requested end time
            adjusted_start_time: Adjusted start time
            adjusted_end_time: Adjusted end time

        Returns:
            FeatureCoverageAnalysis indicating no data
        """
        # All features are missing since there's no data
        feature_coverage = {
            name: FeatureCoverageInfo(
                feature_name=name,
                is_fully_covered=False,
                earliest_timestamp=None,
                latest_timestamp=None,
                record_count=0,
                coverage_percentage=0.0,
                missing_periods=[(requested_start_time, requested_end_time)]
            )
            for name in feature_names
        }

        return FeatureCoverageAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            requested_start_time=requested_start_time,
            requested_end_time=requested_end_time,
            ohlcv_available=False,
            ohlcv_earliest_timestamp=None,
            ohlcv_latest_timestamp=None,
            ohlcv_record_count=0,
            requires_resampling=False,  # No data available at all (not even base timeframe)
            adjusted_start_time=adjusted_start_time,
            adjusted_end_time=adjusted_end_time,
            feature_coverage=feature_coverage,
            existing_features_df=None
        )
