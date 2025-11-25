"""
Feature coverage summary model for efficient coverage checking.

This model represents coverage metadata without fetching actual feature values,
enabling 100-1000x faster coverage analysis.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class FeatureCoverageSummary:
    """
    Lightweight coverage summary for a single feature.

    Contains only aggregate metadata (counts, timestamps) without actual values.
    Designed for efficient coverage checking at scale.
    """

    feature_name: str
    """Business domain feature name (not Feast field names)"""

    total_expected_records: int
    """Total number of timestamps in the requested period"""

    non_null_record_count: int
    """Number of records with non-null values"""

    null_record_count: int
    """Number of records with null values"""

    earliest_non_null_timestamp: Optional[datetime]
    """Timestamp of earliest non-null record"""

    latest_non_null_timestamp: Optional[datetime]
    """Timestamp of latest non-null record"""

    @property
    def coverage_percentage(self) -> float:
        """Calculate coverage percentage."""
        if self.total_expected_records == 0:
            return 0.0
        return (self.non_null_record_count / self.total_expected_records) * 100.0

    @property
    def is_fully_covered(self) -> bool:
        """Check if feature is fully covered (>99%)."""
        return self.coverage_percentage >= 99.0

    @property
    def is_completely_missing(self) -> bool:
        """Check if feature has no data at all."""
        return self.non_null_record_count == 0

    @property
    def has_partial_coverage(self) -> bool:
        """Check if feature has partial coverage."""
        return 0 < self.non_null_record_count < self.total_expected_records

    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"FeatureCoverageSummary({self.feature_name}: "
            f"{self.non_null_record_count}/{self.total_expected_records} records, "
            f"{self.coverage_percentage:.1f}% coverage, "
            f"range=[{self.earliest_non_null_timestamp} - {self.latest_non_null_timestamp}])"
        )
