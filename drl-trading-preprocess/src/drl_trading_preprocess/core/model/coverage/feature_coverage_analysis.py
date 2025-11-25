"""
Feature coverage analysis result model.

Represents the result of analyzing feature coverage for a given time period,
determining which features need computation and which are already available.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from pandas import DataFrame

from drl_trading_common.core.model.timeframe import Timeframe


@dataclass
class OhlcvAvailability:
    """OHLCV data availability information."""
    available: bool
    record_count: int
    earliest_timestamp: Optional[datetime]
    latest_timestamp: Optional[datetime]


@dataclass
class ComputationPeriod:
    """Period for computation."""
    start_time: datetime
    end_time: datetime


@dataclass
class WarmupPeriod:
    """Period for warmup."""
    start_time: datetime
    end_time: datetime


@dataclass
class FeatureCoverageInfo:
    """Coverage information for a single feature."""

    feature_name: str
    is_fully_covered: bool
    earliest_timestamp: Optional[datetime]
    latest_timestamp: Optional[datetime]
    record_count: int
    coverage_percentage: float
    missing_periods: List[tuple[datetime, datetime]]  # List of (start, end) gaps


@dataclass
class FeatureCoverageAnalysis:
    """
    Complete analysis of feature coverage for a symbol-timeframe-period combination.

    This replaces the previous FeatureExistenceCheckResult with a more comprehensive
    analysis that includes:
    - Individual feature coverage details
    - OHLCV data availability constraints
    - Gap detection for incremental processing
    - Warmup requirements identification
    """

    symbol: str
    timeframe: Timeframe
    requested_start_time: datetime
    requested_end_time: datetime

    # OHLCV data constraints (actual data availability in TimescaleDB)
    ohlcv_available: bool
    ohlcv_earliest_timestamp: Optional[datetime]
    ohlcv_latest_timestamp: Optional[datetime]
    ohlcv_record_count: int

    # Adjusted time period based on OHLCV availability
    adjusted_start_time: datetime
    adjusted_end_time: datetime

    # Feature-specific coverage
    feature_coverage: Dict[str, FeatureCoverageInfo]

    # Resampling flag for cold start scenarios
    # When True, indicates coverage was determined from base timeframe
    # and target timeframe data needs to be resampled before feature computation
    requires_resampling: bool = False

    # Fetched features DataFrame (if any exist)
    existing_features_df: Optional[DataFrame] = None

    def __post_init__(self) -> None:
        """Validate and compute derived properties."""
        if not self.ohlcv_available and self.ohlcv_record_count > 0:
            raise ValueError("OHLCV marked as unavailable but has records")

        if self.adjusted_start_time > self.adjusted_end_time:
            raise ValueError("Adjusted start time cannot be after end time")
