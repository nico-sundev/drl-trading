"""
Feature computation response DTO for preprocess service.

Provides comprehensive feedback about processing results, performance metrics,
and metadata for downstream services.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from drl_trading_common.base.base_schema import BaseSchema
from drl_trading_common.model.timeframe import Timeframe
from pydantic import Field


@dataclass
class FeatureProcessingStats(BaseSchema):
    """Statistics for feature processing operations."""

    features_requested: int
    features_computed: int
    features_skipped: int  # Already existed
    features_failed: int

    # Timeframe-specific stats
    timeframes_processed: Dict[str, int] = Field(default_factory=dict)

    # Performance metrics
    processing_duration_ms: int
    data_points_processed: int
    computation_rate_per_second: float

    # Memory usage
    peak_memory_mb: Optional[float] = None
    avg_memory_mb: Optional[float] = None

    def get_success_rate(self) -> float:
        """Calculate feature computation success rate."""
        if self.features_requested == 0:
            return 0.0
        return (self.features_computed / self.features_requested) * 100.0

    def get_skip_rate(self) -> float:
        """Calculate feature skip rate (already existed)."""
        if self.features_requested == 0:
            return 0.0
        return (self.features_skipped / self.features_requested) * 100.0


@dataclass
class FeatureStoreMetadata(BaseSchema):
    """Metadata about feature store operations."""

    offline_store_paths: List[str] = Field(default_factory=list)
    online_store_updated: bool = False
    materialization_completed: bool = False

    # Feature view information
    feature_views_created: List[str] = Field(default_factory=list)
    feature_views_updated: List[str] = Field(default_factory=list)

    # Versioning information
    feature_schema_version: Optional[str] = None
    computation_timestamp: Optional[datetime] = None


@dataclass
class FeatureComputationResponse(BaseSchema):
    """
    Response from feature computation processing.

    Provides comprehensive feedback about processing results, performance,
    and metadata for downstream services.
    """

    # Request correlation
    request_id: str
    symbol: str
    processing_context: str

    # Processing outcome
    success: bool
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, str]] = None

    # Processing statistics
    stats: FeatureProcessingStats

    # Feature store metadata
    feature_store_metadata: FeatureStoreMetadata

    # Timing information
    started_at: datetime
    completed_at: datetime

    # Data quality metrics
    data_quality_issues: List[str] = Field(default_factory=list)
    missing_data_periods: List[str] = Field(default_factory=list)

    # Feature-specific results
    feature_computation_details: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    def get_processing_duration_seconds(self) -> float:
        """Get total processing duration in seconds."""
        return (self.completed_at - self.started_at).total_seconds()

    def get_throughput_features_per_second(self) -> float:
        """Calculate feature computation throughput."""
        duration = self.get_processing_duration_seconds()
        if duration == 0:
            return 0.0
        return self.stats.features_computed / duration

    def has_data_quality_issues(self) -> bool:
        """Check if there were data quality issues."""
        return len(self.data_quality_issues) > 0 or len(self.missing_data_periods) > 0

    def get_summary_message(self) -> str:
        """Get human-readable summary of processing results."""
        if not self.success:
            return f"Failed: {self.error_message}"

        success_rate = self.stats.get_success_rate()
        skip_rate = self.stats.get_skip_rate()

        return (
            f"Processed {self.stats.features_requested} features for {self.symbol} "
            f"({self.stats.features_computed} computed, {self.stats.features_skipped} skipped). "
            f"Success rate: {success_rate:.1f}%, Skip rate: {skip_rate:.1f}%. "
            f"Duration: {self.get_processing_duration_seconds():.2f}s"
        )


@dataclass
class FeatureExistenceCheckResult(BaseSchema):
    """Result of checking which features already exist in the feature store."""

    symbol: str
    timeframe: Timeframe
    existing_features: List[str] = Field(default_factory=list)
    missing_features: List[str] = Field(default_factory=list)

    # Temporal coverage information
    latest_timestamp: Optional[datetime] = None
    earliest_timestamp: Optional[datetime] = None
    coverage_gaps: List[Dict[str, datetime]] = Field(default_factory=list)

    def get_completion_rate(self, requested_features: List[str]) -> float:
        """Calculate what percentage of requested features already exist."""
        if not requested_features:
            return 0.0

        existing_count = len(set(self.existing_features) & set(requested_features))
        return (existing_count / len(requested_features)) * 100.0

    def get_missing_feature_names(self, requested_features: List[str]) -> List[str]:
        """Get list of features that need to be computed."""
        requested_set = set(requested_features)
        existing_set = set(self.existing_features)
        return list(requested_set - existing_set)

    def has_temporal_gaps(self) -> bool:
        """Check if there are temporal gaps in existing features."""
        return len(self.coverage_gaps) > 0
