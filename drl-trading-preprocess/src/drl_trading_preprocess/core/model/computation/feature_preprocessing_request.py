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
from drl_trading_common.model.feature_config_version_info import (
    FeatureConfigVersionInfo,
)
from drl_trading_common.model.timeframe import Timeframe
from pydantic import Field, field_validator


@dataclass
class FeaturePreprocessingRequest(BaseSchema):
    """
    DTO for feature preprocessing requests from ingest service.

    Encapsulates all information needed for dynamic feature processing
    without coupling to static configuration.
    """

    # Core processing parameters
    symbol: str
    base_timeframe: Timeframe
    target_timeframes: List[Timeframe]
    feature_config_version_info: (
        FeatureConfigVersionInfo  # Contains feature config and timeframes
    )
    start_time: datetime
    end_time: datetime
    request_id: str

    # Processing mode configuration
    force_recompute: bool = Field(
        default=False, description="Override existing features"
    )
    incremental_mode: bool = Field(default=True, description="vs full recomputation")

    # Context for different use cases
    processing_context: str = Field(
        default="training", description="training | inference | backfill"
    )

    # Feature store configuration
    skip_existing_features: bool = Field(
        default=True, description="Skip features that already exist in Feast"
    )
    materialize_online: bool = Field(
        default=False, description="Push to online store after computation"
    )

    # Performance tuning
    batch_size: Optional[int] = Field(default=None, description="Processing batch size")
    parallel_processing: bool = Field(
        default=True, description="Enable parallel feature computation"
    )

    @field_validator("symbol")
    def validate_symbol(cls, v: str) -> str:
        """Validate symbol format."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Symbol cannot be empty")
        return v.upper().strip()

    @field_validator("feature_config_version_info")
    def validate_feature_config_version_info(
        cls, v: FeatureConfigVersionInfo
    ) -> FeatureConfigVersionInfo:
        """Validate feature configuration version info."""
        if not v.feature_definitions:
            raise ValueError("Feature definitions cannot be empty in configuration")

        # Check for enabled features
        enabled_features = [f for f in v.feature_definitions if f.get("enabled", False)]
        if not enabled_features:
            raise ValueError("At least one feature must be enabled in configuration")

        return v

    @field_validator("end_time", mode="after")
    def validate_time_range(
        cls: type["FeaturePreprocessingRequest"], v: datetime, info: object
    ) -> datetime:
        """Validate time range."""
        start_time = getattr(info, "data", {}).get("start_time") if hasattr(info, "data") else None
        if start_time and v <= start_time:
            raise ValueError("End time must be after start time")
        return v

    @field_validator("target_timeframes", mode="before")
    def validate_target_timeframes(cls, v: List[Timeframe]) -> List[Timeframe]:
        """Validate target timeframes are not empty."""
        if not v:
            raise ValueError("Target timeframes cannot be empty")
        return v

    def __post_init__(self) -> None:
        """Cross-field validation for target_timeframes and base_timeframe."""
        if self.base_timeframe and self.target_timeframes:
            for tf in self.target_timeframes:
                if tf.to_minutes() <= self.base_timeframe.to_minutes():
                    raise ValueError(
                        f"Target timeframe {tf.value} must be higher than base timeframe {self.base_timeframe.value}"
                    )

    def get_enabled_features(self) -> List[FeatureDefinition]:
        """Get only enabled feature definitions as domain objects."""
        feature_definitions = []
        for feature_dict in self.feature_config_version_info.feature_definitions:
            if feature_dict.get("enabled", False):
                # Convert dict to FeatureDefinition
                feature_def = FeatureDefinition(
                    name=feature_dict["name"],
                    enabled=feature_dict["enabled"],
                    derivatives=feature_dict.get("derivatives", []),
                    parameter_sets=feature_dict.get("parameter_sets", []),
                    parsed_parameter_sets={},  # Will be populated during parsing
                )
                feature_definitions.append(feature_def)
        return feature_definitions

    def get_all_feature_definitions(self) -> List[FeatureDefinition]:
        """Get all feature definitions as domain objects (enabled and disabled)."""
        feature_definitions = []
        for feature_dict in self.feature_config_version_info.feature_definitions:
            # Convert dict to FeatureDefinition
            feature_def = FeatureDefinition(
                name=feature_dict["name"],
                enabled=feature_dict["enabled"],
                derivatives=feature_dict.get("derivatives", []),
                parameter_sets=feature_dict.get("parameter_sets", []),
                parsed_parameter_sets={},  # Will be populated during parsing
            )
            feature_definitions.append(feature_def)
        return feature_definitions

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
