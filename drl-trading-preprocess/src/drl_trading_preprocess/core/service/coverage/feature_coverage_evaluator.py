"""
Feature coverage evaluator for computing derived metrics from coverage analysis.

This evaluator provides operations on FeatureCoverageAnalysis objects to extract
insights, metrics, and decision-support data without embedding logic in the data models.
"""

import logging
from datetime import timedelta
from typing import List, Optional

from injector import inject

from drl_trading_preprocess.core.model.coverage.feature_coverage_analysis import (
    ComputationPeriod,
    FeatureCoverageAnalysis,
    FeatureCoverageInfo,
    WarmupPeriod,
)

logger = logging.getLogger(__name__)


@inject
class FeatureCoverageEvaluator:
    """
    Evaluates feature coverage analysis to provide derived metrics and insights.

    This class contains all the logic for computing properties, summaries, and
    decision-support data from FeatureCoverageAnalysis instances, keeping the
    data models pure and focused on data representation.
    """

    def __init__(self) -> None:
        """Initialize the feature coverage evaluator."""
        logger.info("FeatureCoverageEvaluator initialized")

    def needs_computation(self, info: FeatureCoverageInfo) -> bool:
        """Check if a feature needs computation."""
        return not info.is_fully_covered or info.record_count == 0

    def needs_warmup(self, info: FeatureCoverageInfo) -> bool:
        """Check if a feature needs warmup."""
        return len(info.missing_periods) > 0

    def get_fully_covered_features(
        self, analysis: FeatureCoverageAnalysis
    ) -> List[str]:
        """Get list of feature names that are fully covered."""
        return [
            name
            for name, info in analysis.feature_coverage.items()
            if info.is_fully_covered
        ]

    def get_partially_covered_features(
        self, analysis: FeatureCoverageAnalysis
    ) -> List[str]:
        """Get list of feature names that are partially covered."""
        return [
            name
            for name, info in analysis.feature_coverage.items()
            if not info.is_fully_covered and info.record_count > 0
        ]

    def get_missing_features(self, analysis: FeatureCoverageAnalysis) -> List[str]:
        """Get list of feature names that don't exist at all."""
        return [
            name
            for name, info in analysis.feature_coverage.items()
            if info.record_count == 0
        ]

    def get_features_needing_computation(
        self, analysis: FeatureCoverageAnalysis
    ) -> List[str]:
        """Get all features that need any computation."""
        return [
            name
            for name, info in analysis.feature_coverage.items()
            if self.needs_computation(info)
        ]

    def get_features_needing_warmup(
        self, analysis: FeatureCoverageAnalysis
    ) -> List[str]:
        """Get features that need warmup before computation."""
        return [
            name
            for name, info in analysis.feature_coverage.items()
            if self.needs_warmup(info)
        ]

    def get_overall_coverage_percentage(
        self, analysis: FeatureCoverageAnalysis
    ) -> float:
        """Calculate overall coverage percentage across all features."""
        if not analysis.feature_coverage:
            return 0.0

        total_coverage = sum(
            info.coverage_percentage for info in analysis.feature_coverage.values()
        )
        return total_coverage / len(analysis.feature_coverage)

    def is_ohlcv_constrained(self, analysis: FeatureCoverageAnalysis) -> bool:
        """Check if requested period exceeds available OHLCV data."""
        return (
            analysis.adjusted_start_time != analysis.requested_start_time
            or analysis.adjusted_end_time != analysis.requested_end_time
        )

    def get_computation_period(
        self, analysis: FeatureCoverageAnalysis
    ) -> ComputationPeriod:
        """Get the actual period for which computation should occur."""
        return ComputationPeriod(
            start_time=analysis.adjusted_start_time, end_time=analysis.adjusted_end_time
        )

    def get_warmup_period(
        self, analysis: FeatureCoverageAnalysis, warmup_candles: int = 500
    ) -> Optional[WarmupPeriod]:
        """
        Calculate the warmup period needed before the computation period.

        Warmup is skipped if:
        1. No features need warmup
        2. Any feature needing warmup has no existing records (not backed by existing features)
        3. Total available candles for the timeframe is below warmup_candles
        4. Computation period starts within the covered period (no gap to fill)

        The warmup end always points to the last available feature timestamp for features needing warmup.
        """
        if not self.get_features_needing_warmup(analysis):
            return None

        # Check if any feature needing warmup has no existing records
        if any(
            info.record_count == 0
            for info in analysis.feature_coverage.values()
            if self.needs_warmup(info)
        ):
            return None

        timeframe_minutes = analysis.timeframe.to_minutes()

        # Check if total available candles is sufficient
        if analysis.ohlcv_latest_timestamp and analysis.ohlcv_earliest_timestamp:
            total_available_candles = (
                analysis.ohlcv_latest_timestamp - analysis.ohlcv_earliest_timestamp
            ) / timedelta(minutes=timeframe_minutes)
            if total_available_candles < warmup_candles:
                return None
        else:
            # No OHLCV data available at all
            return None

        # Find the last available feature timestamp for features needing warmup
        latest_timestamps = [
            info.latest_timestamp
            for info in analysis.feature_coverage.values()
            if self.needs_warmup(info) and info.latest_timestamp is not None
        ]
        if not latest_timestamps:
            return None

        warmup_end = min(latest_timestamps)

        # If computation period starts within covered period, no warmup needed
        if analysis.adjusted_start_time <= warmup_end:
            return None

        warmup_duration = timedelta(minutes=warmup_candles * timeframe_minutes)
        warmup_start = warmup_end - warmup_duration

        # Constrain warmup start by OHLCV availability
        if (
            analysis.ohlcv_earliest_timestamp
            and warmup_start < analysis.ohlcv_earliest_timestamp
        ):
            warmup_start = analysis.ohlcv_earliest_timestamp

        # If no historical data exists for warmup, skip it
        if warmup_start >= warmup_end:
            return None

        return WarmupPeriod(start_time=warmup_start, end_time=warmup_end)

    def get_summary_message(self, analysis: FeatureCoverageAnalysis) -> str:
        """Get human-readable summary of the analysis."""
        return (
            f"Feature coverage for {analysis.symbol} {analysis.timeframe.value}: "
            f"{len(self.get_fully_covered_features(analysis))}/{len(analysis.feature_coverage)} fully covered "
            f"({self.get_overall_coverage_percentage(analysis):.1f}%), "
            f"{len(self.get_missing_features(analysis))} missing, "
            f"{len(self.get_partially_covered_features(analysis))} partial. "
            f"OHLCV: {analysis.ohlcv_record_count} records "
            f"[{analysis.ohlcv_earliest_timestamp} - {analysis.ohlcv_latest_timestamp}]"
        )
