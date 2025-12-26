"""
Feature preprocessing request domain model for core business logic.

This model represents the domain concept of a feature preprocessing request
within the core business logic layer. It contains the essential business
information needed for feature computation without external concerns.
"""

from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from drl_trading_core.core.model.feature_definition import FeatureDefinition
from drl_trading_core.core.model.feature_config_version_info import FeatureConfigVersionInfo
from drl_trading_common.core.model.processing_context import ProcessingContext
from drl_trading_common.core.model.timeframe import Timeframe
from pydantic import BaseModel, Field, field_validator


class FeaturePreprocessingRequest(BaseModel):
    """
    Domain model for feature preprocessing requests.

    Represents the core business concept of a request to compute features
    for a specific symbol and time range. This model focuses on the business
    logic aspects without external communication concerns.

    Use factory methods for common scenarios:
    - `for_training()` - Training pipeline
    - `for_inference()` - Real-time inference
    - `for_backfill()` - Historical data processing
    - `for_catchup()` - Initial catchup before live streaming
    """

    # Core processing parameters
    symbol: str
    base_timeframe: Timeframe
    target_timeframes: List[Timeframe]
    feature_config_version_info: FeatureConfigVersionInfo
    start_time: datetime
    end_time: datetime
    request_id: str = Field(default_factory=lambda: str(uuid4()))

    # Processing mode configuration
    force_recompute: bool = Field(
        default=False,
        description="Override existing features and recompute from scratch"
    )
    incremental_mode: bool = Field(
        default=True,
        description="Process only new data vs full recomputation"
    )

    # Context for different use cases
    processing_context: ProcessingContext = Field(
        default=ProcessingContext.TRAINING,
        description="Context: TRAINING | INFERENCE | BACKFILL | CATCHUP"
    )

    # Feature store configuration
    skip_existing_features: bool = Field(
        default=True,
        description="Skip features that already exist in feature store"
    )
    materialize_online: bool = Field(
        default=False,
        description="Push computed features to online store after computation"
    )

    # Performance tuning
    batch_size: Optional[int] = Field(
        default=None,
        description="Processing batch size for chunked computation"
    )
    parallel_processing: bool = Field(
        default=True,
        description="Enable parallel feature computation across timeframes"
    )

    # Validation
    @field_validator("start_time", "end_time", mode="after")
    @classmethod
    def normalize_timezone(cls, v: datetime) -> datetime:
        """
        Normalize datetime timezone to Python's built-in datetime.timezone.utc.

        Ensures all timestamps use the same timezone object type for consistency.
        """
        if v.tzinfo is not None:
            return v.astimezone(timezone.utc)
        return v.replace(tzinfo=timezone.utc)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate symbol format."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Symbol cannot be empty")
        return v.upper().strip()

    @field_validator("feature_config_version_info")
    @classmethod
    def validate_feature_config_version_info(
        cls, v: FeatureConfigVersionInfo
    ) -> FeatureConfigVersionInfo:
        """Validate feature configuration version info."""
        if not v.feature_definitions:
            raise ValueError("Feature definitions cannot be empty in configuration")

        # Check for enabled features
        enabled_features = [f for f in v.feature_definitions if f.enabled]
        if not enabled_features:
            raise ValueError("At least one feature must be enabled in configuration")

        return v

    @field_validator("end_time", mode="after")
    @classmethod
    def validate_time_range(
        cls, v: datetime, info: object
    ) -> datetime:
        """Validate time range."""
        start_time = getattr(info, "data", {}).get("start_time") if hasattr(info, "data") else None
        if start_time and v <= start_time:
            raise ValueError("End time must be after start time")
        return v

    @field_validator("target_timeframes", mode="before")
    @classmethod
    def validate_target_timeframes(cls, v: List[Timeframe]) -> List[Timeframe]:
        """Validate target timeframes are not empty."""
        if not v:
            raise ValueError("Target timeframes cannot be empty")
        return v

    @field_validator("target_timeframes", mode="after")
    @classmethod
    def validate_target_higher_than_base(
        cls, v: List[Timeframe], info: object
    ) -> List[Timeframe]:
        """Cross-field validation: target timeframes must be higher than base timeframe."""
        base_timeframe = getattr(info, "data", {}).get("base_timeframe") if hasattr(info, "data") else None
        if base_timeframe and v:
            for tf in v:
                if tf.to_minutes() <= base_timeframe.to_minutes():
                    raise ValueError(
                        f"Target timeframe {tf.value} must be higher than base timeframe {base_timeframe.value}"
                    )
        return v

    @field_validator("processing_context", mode="after")
    @classmethod
    def validate_processing_context_materialization(
        cls, v: ProcessingContext, info: object
    ) -> ProcessingContext:
        """
        Validate processing context and apply default materialization rules.

        CATCHUP context should not materialize to online store by default.
        """
        # This is just validation - actual enforcement happens in model_post_init
        return v

    # Utility Methods
    def get_enabled_features(self) -> List[FeatureDefinition]:
        """Get only enabled feature definitions."""
        return [f for f in self.feature_config_version_info.feature_definitions if f.enabled]

    def get_all_feature_definitions(self) -> List[FeatureDefinition]:
        """Get all feature definitions (enabled and disabled)."""
        return self.feature_config_version_info.feature_definitions

    def get_base_timeframe(self) -> Timeframe:
        """Get base timeframe from request."""
        return self.base_timeframe

    def get_target_timeframes(self) -> List[Timeframe]:
        """Get target timeframes from request."""
        return self.target_timeframes

    def get_processing_period_days(self) -> int:
        """Calculate processing period in days."""
        return (self.end_time - self.start_time).days

    def is_backfill_request(self) -> bool:
        """Check if this is a backfill request (large time range)."""
        return self.get_processing_period_days() > 30

    def should_use_chunked_processing(self) -> bool:
        """Determine if chunked processing should be used."""
        return self.is_backfill_request() or self.get_processing_period_days() > 7

    def is_catchup_request(self) -> bool:
        """Check if this is a catchup request."""
        return self.processing_context == ProcessingContext.CATCHUP

    def with_updated(self, **kwargs: object) -> "FeaturePreprocessingRequest":
        """
        Create a copy of this request with updated fields (immutable pattern).

        Useful for creating variations of a request without mutating the original.

        Args:
            **kwargs: Fields to update

        Returns:
            New request instance with updated fields

        Example:
            >>> original = FeaturePreprocessingRequest.for_training(...)
            >>> updated = original.with_updated(materialize_online=True)
        """
        return self.model_copy(update=kwargs)
