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
from typing import Any, Dict, List

import pandas as pd
from injector import inject
from pandas import DataFrame

from drl_trading_core.core.model.feature_config_version_info import FeatureConfigVersionInfo
from drl_trading_core.core.service.feature_manager import FeatureManager
from drl_trading_common.core.model.timeframe import Timeframe
from drl_trading_core.core.dto.feature_service_metadata import FeatureServiceMetadata
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


def _normalize_timestamp(ts: Any) -> datetime:
    """
    Normalize any timestamp-like value to a UTC-aware Python datetime.

    Handles numpy datetime64, pandas Timestamp (with/without tz), and Python datetime.
    This ensures consistent types when comparing timestamps from different sources
    (Feast DataFrames, pd.date_range, request parameters).

    Args:
        ts: Any timestamp-like value

    Returns:
        UTC-aware Python datetime
    """
    pd_ts = pd.Timestamp(ts)
    if pd_ts.tz is None:
        pd_ts = pd_ts.tz_localize('UTC')
    return pd_ts.to_pydatetime()


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
        feature_manager: FeatureManager,
    ) -> None:
        """
        Initialize the feature coverage analyzer.

        Args:
            feature_store_fetch_port: Port for fetching features from Feast
            market_data_reader: Port for checking OHLCV data availability
            feature_coverage_evaluator: Evaluator for coverage analysis operations
            feature_manager: Manager for accessing feature metadata
        """
        self.feature_store_fetch_port = feature_store_fetch_port
        self.market_data_reader = market_data_reader
        self.feature_coverage_evaluator = feature_coverage_evaluator
        self.feature_manager = feature_manager
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
        feature_service_metadata_list: List[FeatureServiceMetadata],
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

        # Step 1b: Also check base timeframe to determine resampling needs
        base_availability = self._check_ohlcv_availability(
            symbol, base_timeframe, requested_start_time, requested_end_time
        )

        # Step 1c: Determine if resampling is needed
        if not ohlcv_availability.available:
            # Cold start: No target timeframe data exists
            if base_availability.available:
                requires_resampling = True
                ohlcv_availability = base_availability
                logger.info(
                    f"No data in target timeframe {timeframe.value}, "
                    f"checking base timeframe {base_timeframe.value} for cold start"
                )
                logger.info(
                    f"Found {base_availability.record_count} records in base timeframe {base_timeframe.value}. "
                    f"Target timeframe {timeframe.value} will be resampled from base."
                )
        else:
            # Target data exists - check if base has newer data (incremental resampling)
            if base_availability.available and base_availability.latest_timestamp and ohlcv_availability.latest_timestamp:
                # Calculate the minimum time difference that would constitute new data to resample
                # For a 5m timeframe, we need at least 5 minutes of new 1m data
                timeframe_minutes = timeframe.to_minutes()
                time_diff_seconds = (base_availability.latest_timestamp - ohlcv_availability.latest_timestamp).total_seconds()
                time_diff_minutes = time_diff_seconds / 60.0

                if time_diff_minutes >= timeframe_minutes:
                    requires_resampling = True
                    logger.info(
                        f"Incremental resampling needed: Base timeframe {base_timeframe.value} has data "
                        f"up to {base_availability.latest_timestamp}, target timeframe {timeframe.value} "
                        f"only has data up to {ohlcv_availability.latest_timestamp} "
                        f"({time_diff_minutes:.1f} minutes gap >= {timeframe_minutes} minutes threshold)"
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
            feature_service_metadata_list=feature_service_metadata_list
        )

        # Step 4: Analyze coverage for each feature
        feature_coverage = self._analyze_individual_features(
            feature_names=feature_names,
            existing_features_df=existing_features_df,
            adjusted_start_time=adjusted_start_time,
            adjusted_end_time=adjusted_end_time,
            timeframe=timeframe,
            feature_service_metadata_list=feature_service_metadata_list
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
        feature_service_metadata_list: List[FeatureServiceMetadata],
    ) -> DataFrame:
        """
        Batch fetch all features from Feast for the given period.

        Fetches features separately for each feature role (observation, reward)
        and merges the resulting dataframes.

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

            # Fetch features for each role separately
            feature_dfs = []
            for feature_service_metadata in feature_service_metadata_list:
                try:
                    # Fetch features from Feast for this role
                    role_features_df = self.feature_store_fetch_port.get_offline(
                        feature_service_metadata, pd.Series(timestamps)
                    )

                    if not role_features_df.empty:
                        feature_dfs.append(role_features_df)
                        logger.info(
                            f"Fetched {len(role_features_df)} records with {len(role_features_df.columns)} columns "
                            f"from Feast for role {feature_service_metadata.feature_service_role.value}"
                        )
                    else:
                        logger.info(f"No existing features found in Feast for {symbol} {timeframe.value} role {feature_service_metadata.feature_service_role.value}")

                except Exception as e:
                    logger.error(f"Error fetching features for role {feature_service_metadata.feature_service_role.value}: {e}")
                    # Continue with other roles

            # Merge all feature dataframes
            if not feature_dfs:
                logger.info(f"No existing features found in Feast for {symbol} {timeframe.value} across all roles")
                return DataFrame()

            # Merge dataframes on index (time), using outer join to handle different time ranges
            merged_df = pd.concat(feature_dfs, axis=1, join='outer')

            logger.info(
                f"Merged features: {len(merged_df)} records with {len(merged_df.columns)} columns "
                f"from {len(feature_dfs)} roles"
            )

            return merged_df

        except Exception as e:
            logger.error(f"Error fetching features from Feast: {e}")
            return DataFrame()

    def _build_feature_column_mapping(
        self,
        feature_service_metadata_list: List[FeatureServiceMetadata] | None
    ) -> Dict[str, List[str]]:
        """
        Build mapping from base feature names to actual Feast column names.

        Extracts FeatureMetadata from service metadata and constructs the exact
        column names by combining feature metadata string (with config hash) and
        sub-feature names.

        Args:
            feature_service_metadata_list: Service metadata containing feature view metadata

        Returns:
            Dict mapping base feature names to list of actual column names
        """
        feature_name_to_columns: Dict[str, List[str]] = {}

        if not feature_service_metadata_list:
            return feature_name_to_columns

        for service_metadata in feature_service_metadata_list:
            for view_metadata in service_metadata.feature_view_metadata_list:
                feature_metadata = view_metadata.feature_metadata
                base_name = feature_metadata.feature_name

                if base_name not in feature_name_to_columns:
                    feature_name_to_columns[base_name] = []

                # Build exact column names: {full_feature_name}_{sub_feature_name}
                # where full_feature_name from __str__() includes config hash
                full_feature_name = str(feature_metadata)
                feature_name_to_columns[base_name].append(full_feature_name)

                for sub_feature_name in feature_metadata.sub_feature_names:
                    column_name = f"{full_feature_name}_{sub_feature_name}"
                    feature_name_to_columns[base_name].append(column_name)

        return feature_name_to_columns

    def _find_matching_columns(
        self,
        feature_name: str,
        expected_column_names: List[str],
        existing_features_df: DataFrame
    ) -> List[str]:
        """
        Find columns in dataframe matching a feature name.

        First tries metadata-based matching, then falls back to prefix matching.

        Args:
            feature_name: Base feature name to search for
            expected_column_names: Expected column names from metadata
            existing_features_df: DataFrame with existing features

        Returns:
            List of matching column names
        """
        # Find matching columns from metadata
        matching_cols = [col for col in expected_column_names if col in existing_features_df.columns]

        # Fallback: if no metadata-based matches found, try simple matching
        if not matching_cols:
            # Check for exact match first
            if feature_name in existing_features_df.columns:
                matching_cols.append(feature_name)

            # Then check for parametrized versions
            for col in existing_features_df.columns:
                if col.startswith(f"{feature_name}_") and col not in matching_cols:
                    matching_cols.append(col)

        return matching_cols

    def _analyze_feature_column_coverage(
        self,
        feature_name: str,
        col_name: str,
        existing_features_df: DataFrame,
        expected_timestamps: pd.DatetimeIndex,
        expected_count: int
    ) -> FeatureCoverageInfo | None:
        """
        Analyze coverage for a single feature column.

        Args:
            feature_name: Base feature name
            col_name: Actual column name in dataframe
            existing_features_df: DataFrame with existing features
            expected_timestamps: Expected timestamps for full coverage
            expected_count: Expected number of records

        Returns:
            FeatureCoverageInfo if column has data, None if empty
        """
        feature_series = existing_features_df[col_name]

        # Count non-null records for coverage percentage
        non_null_series = feature_series.dropna()
        record_count = len(non_null_series)

        # Check if we have all expected timestamps (even if some values are NaN)
        total_records = len(feature_series)

        if total_records == 0:
            # Feature column exists but has no data at all in the time range
            return None

        # Feature has been computed (records exist in time range)
        # For earliest/latest timestamps, use non-null values for accurate bounds
        earliest_idx = non_null_series.index.min() if record_count > 0 else feature_series.index.min()
        latest_idx = non_null_series.index.max() if record_count > 0 else feature_series.index.max()
        earliest_ts = _normalize_timestamp(earliest_idx)
        latest_ts = _normalize_timestamp(latest_idx)

        # Identify missing periods (gaps in the data where timestamps have NaN values)
        missing_periods = self._identify_missing_periods(
            feature_series,
            expected_timestamps
        )

        # Calculate coverage based on non-null values
        coverage_pct = (record_count / expected_count * 100.0) if expected_count > 0 else 0.0

        # Feature is fully covered if there are no missing periods (timestamp gaps)
        is_fully_covered = len(missing_periods) == 0

        return FeatureCoverageInfo(
            feature_name=feature_name,
            is_fully_covered=is_fully_covered,
            earliest_timestamp=earliest_ts,
            latest_timestamp=latest_ts,
            record_count=record_count,  # Report non-null count for metrics
            coverage_percentage=coverage_pct,
            missing_periods=missing_periods
        )

    def _analyze_individual_features(
        self,
        feature_names: List[str],
        existing_features_df: DataFrame,
        adjusted_start_time: datetime,
        adjusted_end_time: datetime,
        timeframe: Timeframe,
        feature_service_metadata_list: List[FeatureServiceMetadata] | None = None,
    ) -> Dict[str, FeatureCoverageInfo]:
        """
        Analyze coverage for each individual feature.

        Maps base feature names to their actual Feast column names by extracting
        the full feature names (with config hashes) from FeatureMetadata instances
        in the passed service metadata.

        For each feature, the actual column name is: {feature_metadata}_{sub_feature_name}
        where feature_metadata.__str__() includes the config hash.

        Args:
            feature_names: List of base feature names to analyze
            existing_features_df: DataFrame with existing features
            adjusted_start_time: Start of analysis period
            adjusted_end_time: End of analysis period
            timeframe: Timeframe
            feature_service_metadata_list: Service metadata containing feature view metadata

        Returns:
            Dict mapping feature names to coverage info
        """
        feature_coverage = {}

        # Build mapping from base feature name to actual Feast column names
        feature_name_to_columns = self._build_feature_column_mapping(feature_service_metadata_list)

        # Calculate expected number of records for the period
        expected_timestamps = pd.date_range(
            start=adjusted_start_time,
            end=adjusted_end_time,
            freq=timeframe.to_pandas_freq()
        )
        expected_count = len(expected_timestamps)

        for feature_name in feature_names:
            # Get the list of expected column names for this feature from metadata
            expected_column_names = feature_name_to_columns.get(feature_name, [])

            # Find matching columns in existing dataframe
            matching_cols = self._find_matching_columns(
                feature_name, expected_column_names, existing_features_df
            )

            if existing_features_df.empty or not matching_cols:
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
                # Analyze each matching column and take the best coverage
                best_coverage = None
                for col_name in matching_cols:
                    current_coverage = self._analyze_feature_column_coverage(
                        feature_name, col_name, existing_features_df,
                        expected_timestamps, expected_count
                    )

                    if current_coverage is None:
                        continue  # Column was empty

                    # Keep the best coverage (prefer fully covered, then highest percentage)
                    if best_coverage is None or \
                       (current_coverage.is_fully_covered and not best_coverage.is_fully_covered) or \
                       (current_coverage.is_fully_covered == best_coverage.is_fully_covered and
                        current_coverage.coverage_percentage > best_coverage.coverage_percentage):
                        best_coverage = current_coverage

                # Use best coverage found, or mark as missing if no valid columns
                if best_coverage is not None:
                    feature_coverage[feature_name] = best_coverage
                else:
                    # All matching columns were empty
                    feature_coverage[feature_name] = FeatureCoverageInfo(
                        feature_name=feature_name,
                        is_fully_covered=False,
                        earliest_timestamp=None,
                        latest_timestamp=None,
                        record_count=0,
                        coverage_percentage=0.0,
                        missing_periods=[(adjusted_start_time, adjusted_end_time)]
                    )

        return feature_coverage

    def _get_expected_interval_seconds(self, expected_timestamps: pd.DatetimeIndex) -> float:
        """
        Get the expected interval between timestamps in seconds.

        Args:
            expected_timestamps: DatetimeIndex with expected timestamps

        Returns:
            Expected interval in seconds, defaults to 3600 (1 hour) if cannot determine
        """
        if expected_timestamps.freq is not None and hasattr(expected_timestamps.freq, 'nanos'):
            return expected_timestamps.freq.nanos / 1e9
        if len(expected_timestamps) > 1:
            return (expected_timestamps[1] - expected_timestamps[0]).total_seconds()
        return 3600.0  # Default 1 hour

    def _exclude_warmup_nulls(
        self,
        null_timestamps: pd.DatetimeIndex,
        feature_series: pd.Series,
        null_mask: pd.Series
    ) -> pd.DatetimeIndex:
        """
        Exclude leading NaN values that represent indicator warmup periods.

        Small leading NaN sequences (< 5% of data) are considered warmup periods
        for indicators like RSI, EMA that need initialization data.

        Args:
            null_timestamps: DatetimeIndex of null value timestamps
            feature_series: Original feature series
            null_mask: Boolean mask of null values

        Returns:
            DatetimeIndex with warmup nulls excluded
        """
        non_null_mask = ~null_mask
        if not non_null_mask.any():
            return null_timestamps

        first_valid_ts = pd.Timestamp(feature_series[non_null_mask].index.min())
        leading_nulls = null_timestamps[null_timestamps < first_valid_ts]

        # Exclude if leading NaNs are < 5% of total records (typical warmup)
        warmup_threshold = len(feature_series) * 0.05
        if 0 < len(leading_nulls) < warmup_threshold:
            return null_timestamps[null_timestamps >= first_valid_ts]

        return null_timestamps

    def _identify_missing_periods(
        self,
        feature_series: pd.Series,
        expected_timestamps: pd.DatetimeIndex
    ) -> List[tuple[datetime, datetime]]:
        """
        Identify gaps/missing periods in feature data.

        Note: Small leading NaN sequences (< 5% of data) are considered warmup periods
        and NOT counted as missing. This handles indicators like RSI, EMA that need
        warmup periods. Larger gaps are considered true missing data.

        Args:
            feature_series: Series with feature data
            expected_timestamps: Expected timestamps for full coverage

        Returns:
            List of (start, end) tuples representing missing periods (excluding small warmup)
        """
        null_mask = feature_series.isnull()
        null_timestamps = feature_series[null_mask].index

        if len(null_timestamps) == 0:
            return []

        # Normalize to pandas DatetimeIndex for consistent behavior
        null_timestamps = pd.DatetimeIndex(null_timestamps)

        # Exclude warmup period nulls
        null_timestamps = self._exclude_warmup_nulls(null_timestamps, feature_series, null_mask)
        if len(null_timestamps) == 0:
            return []

        # Group consecutive null timestamps into periods
        expected_diff = self._get_expected_interval_seconds(expected_timestamps)
        missing_periods: List[tuple[datetime, datetime]] = []
        current_start = None
        previous_ts = None

        for ts in sorted(null_timestamps):
            if current_start is None:
                current_start = ts
            elif previous_ts is not None:
                time_diff = (ts - previous_ts).total_seconds()
                if time_diff > expected_diff * 1.5:  # Gap found
                    missing_periods.append((current_start, previous_ts))
                    current_start = ts
            previous_ts = ts

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
