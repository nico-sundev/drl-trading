"""
Feature coverage analysis result model.

Represents the result of analyzing feature coverage for a given time period,
determining which features need computation and which are already available.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from pandas import DataFrame

from drl_trading_common.model.timeframe import Timeframe


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

    def needs_computation(self) -> bool:
        """Check if this feature needs any computation."""
        return not self.is_fully_covered or self.record_count == 0

    def needs_warmup(self) -> bool:
        """Check if this feature needs warmup (has gaps or missing early data)."""
        return len(self.missing_periods) > 0 or self.coverage_percentage < 100.0


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

    # Fetched features DataFrame (if any exist)
    existing_features_df: Optional[DataFrame] = None

    def __post_init__(self) -> None:
        """Validate and compute derived properties."""
        if not self.ohlcv_available and self.ohlcv_record_count > 0:
            raise ValueError("OHLCV marked as unavailable but has records")

        if self.adjusted_start_time > self.adjusted_end_time:
            raise ValueError("Adjusted start time cannot be after end time")

    @property
    def fully_covered_features(self) -> List[str]:
        """Get list of feature names that are fully covered."""
        return [
            name for name, info in self.feature_coverage.items()
            if info.is_fully_covered
        ]

    @property
    def partially_covered_features(self) -> List[str]:
        """Get list of feature names that are partially covered (need catch-up)."""
        return [
            name for name, info in self.feature_coverage.items()
            if not info.is_fully_covered and info.record_count > 0
        ]

    @property
    def missing_features(self) -> List[str]:
        """Get list of feature names that don't exist at all."""
        return [
            name for name, info in self.feature_coverage.items()
            if info.record_count == 0
        ]

    @property
    def features_needing_computation(self) -> List[str]:
        """Get all features that need any computation (partial or full)."""
        return [
            name for name, info in self.feature_coverage.items()
            if info.needs_computation()
        ]

    @property
    def features_needing_warmup(self) -> List[str]:
        """Get features that need warmup before computation."""
        return [
            name for name, info in self.feature_coverage.items()
            if info.needs_warmup()
        ]

    @property
    def overall_coverage_percentage(self) -> float:
        """Calculate overall coverage percentage across all features."""
        if not self.feature_coverage:
            return 0.0

        total_coverage = sum(info.coverage_percentage for info in self.feature_coverage.values())
        return total_coverage / len(self.feature_coverage)

    @property
    def is_ohlcv_constrained(self) -> bool:
        """Check if requested period exceeds available OHLCV data."""
        return (
            self.adjusted_start_time != self.requested_start_time or
            self.adjusted_end_time != self.requested_end_time
        )

    def get_computation_period(self) -> tuple[datetime, datetime]:
        """
        Get the actual period for which computation should occur.

        Returns the adjusted period constrained by OHLCV availability.
        """
        return (self.adjusted_start_time, self.adjusted_end_time)

    def get_warmup_period(self, warmup_candles: int = 500) -> Optional[tuple[datetime, datetime]]:
        """
        Calculate the warmup period needed before the computation period.

        Args:
            warmup_candles: Number of candles to use for warmup

        Returns:
            Tuple of (warmup_start, warmup_end) or None if no warmup needed
        """
        if not self.features_needing_warmup:
            return None

        from datetime import timedelta
        timeframe_minutes = self.timeframe.to_minutes()
        warmup_duration = timedelta(minutes=warmup_candles * timeframe_minutes)

        # Warmup period ends at the start of computation period
        warmup_end = self.adjusted_start_time
        warmup_start = warmup_end - warmup_duration

        # Constrain warmup start by OHLCV availability
        if self.ohlcv_earliest_timestamp and warmup_start < self.ohlcv_earliest_timestamp:
            warmup_start = self.ohlcv_earliest_timestamp

        return (warmup_start, warmup_end)

    def get_summary_message(self) -> str:
        """Get human-readable summary of the analysis."""
        return (
            f"Feature coverage for {self.symbol} {self.timeframe.value}: "
            f"{len(self.fully_covered_features)}/{len(self.feature_coverage)} fully covered "
            f"({self.overall_coverage_percentage:.1f}%), "
            f"{len(self.missing_features)} missing, "
            f"{len(self.partially_covered_features)} partial. "
            f"OHLCV: {self.ohlcv_record_count} records "
            f"[{self.ohlcv_earliest_timestamp} - {self.ohlcv_latest_timestamp}]"
        )
