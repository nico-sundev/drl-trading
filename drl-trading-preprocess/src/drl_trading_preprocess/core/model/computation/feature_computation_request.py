"""
Feature computation request DTO for ingest → preprocess service communication.

This DTO encapsulates all information needed for dynamic feature processing
without coupling to static configuration.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from drl_trading_common.base.base_schema import BaseSchema
from drl_trading_common.config.feature_config import FeatureDefinition
from drl_trading_common.model.timeframe import Timeframe
from pydantic import Field, validator


@dataclass
class FeatureComputationRequest(BaseSchema):
    """
    DTO for feature computation requests from ingest service.

    Encapsulates all information needed for dynamic feature processing
    without coupling to static configuration.
    """

    # Core processing parameters
    symbol: str
    base_timeframe: Timeframe
    target_timeframes: List[Timeframe]
    feature_definitions: List[FeatureDefinition]  # Dynamic per request
    start_time: datetime
    end_time: datetime
    request_id: str

    # Processing mode configuration
    force_recompute: bool = Field(default=False, description="Override existing features")
    incremental_mode: bool = Field(default=True, description="vs full recomputation")

    # Context for different use cases
    processing_context: str = Field(default="training", description="training | inference | backfill")

    # Feature store configuration
    skip_existing_features: bool = Field(default=True, description="Skip features that already exist in Feast")
    materialize_online: bool = Field(default=False, description="Push to online store after computation")

    # Performance tuning
    batch_size: Optional[int] = Field(default=None, description="Processing batch size")
    parallel_processing: bool = Field(default=True, description="Enable parallel feature computation")

    @validator('symbol')
    def validate_symbol(cls, v: str) -> str:
        """Validate symbol format."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Symbol cannot be empty")
        return v.upper().strip()

    @validator('feature_definitions')
    def validate_feature_definitions(cls, v: List[FeatureDefinition]) -> List[FeatureDefinition]:
        """Validate feature definitions."""
        if not v:
            raise ValueError("Feature definitions cannot be empty")

        # Check for enabled features
        enabled_features = [f for f in v if f.enabled]
        if not enabled_features:
            raise ValueError("At least one feature must be enabled")

        return v

    @validator('end_time')
    def validate_time_range(cls, v: datetime, values: dict) -> datetime:
        """Validate time range."""
        start_time = values.get('start_time')
        if start_time and v <= start_time:
            raise ValueError("End time must be after start time")
        return v

    @validator('target_timeframes')
    def validate_target_timeframes(cls, v: List[Timeframe], values: dict) -> List[Timeframe]:
        """Validate target timeframes against base timeframe."""
        base_timeframe = values.get('base_timeframe')
        if not v:
            raise ValueError("Target timeframes cannot be empty")

        if base_timeframe:
            for tf in v:
                if tf.to_minutes() <= base_timeframe.to_minutes():
                    raise ValueError(f"Target timeframe {tf.value} must be higher than base timeframe {base_timeframe.value}")

        return v

    def get_enabled_features(self) -> List[FeatureDefinition]:
        """Get only enabled feature definitions."""
        return [f for f in self.feature_definitions if f.enabled]

    def get_processing_period_days(self) -> int:
        """Calculate processing period in days."""
        return (self.end_time - self.start_time).days

    def is_backfill_request(self) -> bool:
        """Check if this is a backfill request (large time range)."""
        return self.get_processing_period_days() > 30

    def should_use_chunked_processing(self) -> bool:
        """Determine if chunked processing should be used."""
        return self.is_backfill_request() or self.get_processing_period_days() > 7
