"""
Feature preprocessing request DTO for cross-service communication.

This DTO is used by multiple services (ingest, preprocess, training) to request
feature computation. It encapsulates all information needed for dynamic feature
processing without coupling to static configuration.
"""

from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from drl_trading_common.base.base_schema import BaseSchema
from drl_trading_common.adapter.model.feature_definition import FeatureDefinition
from drl_trading_common.adapter.model.feature_config_version_info import FeatureConfigVersionInfo
from drl_trading_common.adapter.model.timeframe import Timeframe
from pydantic import Field, field_validator


class FeaturePreprocessingRequest(BaseSchema):
    """
    DTO for feature preprocessing requests.

    Encapsulates all information needed for dynamic feature processing
    across multiple services (ingest → preprocess, training → preprocess).

    Use factory methods for common scenarios:
    - `for_training()` - Training pipeline
    - `for_inference()` - Real-time inference
    - `for_backfill()` - Historical data processing
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
    processing_context: str = Field(
        default="training",
        description="Context: training | inference | backfill"
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

        Pydantic's datetime deserialization can return various timezone implementations
        (pytz.UTC, dateutil.tz.UTC, etc.) depending on the source. This validator ensures
        all timestamps use the same timezone object type for consistency across the application.
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

    # Factory Methods for Common Scenarios
    @classmethod
    def for_training(
        cls: type["FeaturePreprocessingRequest"],
        symbol: str,
        target_timeframes: List[Timeframe],
        feature_config: FeatureConfigVersionInfo,
        start_time: datetime,
        end_time: datetime,
        base_timeframe: Timeframe = Timeframe.MINUTE_1,
        **kwargs: object
    ) -> "FeaturePreprocessingRequest":
        """
        Create a preprocessing request for training pipeline.

        Training context defaults:
        - incremental_mode: True (process only new data)
        - skip_existing_features: True (avoid redundant computation)
        - materialize_online: False (training doesn't need online store)
        - parallel_processing: True (maximize throughput)

        Args:
            symbol: Trading symbol (e.g., 'BTCUSD')
            target_timeframes: List of timeframes to compute features for
            feature_config: Feature configuration with version info
            start_time: Start of time range
            end_time: End of time range
            base_timeframe: Base timeframe for resampling (default: 1m)
            **kwargs: Override any other fields

        Returns:
            Configured request for training context
        """
        return cls(
            symbol=symbol,
            base_timeframe=base_timeframe,
            target_timeframes=target_timeframes,
            feature_config_version_info=feature_config,
            start_time=start_time,
            end_time=end_time,
            force_recompute=bool(kwargs.pop('force_recompute', False)),
            incremental_mode=bool(kwargs.pop('incremental_mode', True)),
            processing_context="training",
            skip_existing_features=bool(kwargs.pop('skip_existing_features', True)),
            materialize_online=bool(kwargs.pop('materialize_online', False)),
            batch_size=int(str(kwargs.pop('batch_size'))) if 'batch_size' in kwargs and kwargs['batch_size'] is not None and isinstance(kwargs['batch_size'], (int, str)) else None,
            parallel_processing=bool(kwargs.pop('parallel_processing', True)),
            **kwargs
        )

    @classmethod
    def for_inference(
        cls: type["FeaturePreprocessingRequest"],
        symbol: str,
        target_timeframes: List[Timeframe],
        feature_config: FeatureConfigVersionInfo,
        start_time: datetime,
        end_time: datetime,
        base_timeframe: Timeframe = Timeframe.MINUTE_1,
        **kwargs: object
    ) -> "FeaturePreprocessingRequest":
        """
        Create a preprocessing request for inference pipeline.

        Inference context defaults:
        - incremental_mode: True (real-time data only)
        - skip_existing_features: False (always compute fresh for inference)
        - materialize_online: True (inference needs online store)
        - force_recompute: True (ensure fresh features for trading)
        - parallel_processing: True (low latency)

        Args:
            symbol: Trading symbol (e.g., 'BTCUSD')
            target_timeframes: List of timeframes to compute features for
            feature_config: Feature configuration with version info
            start_time: Start of time range
            end_time: End of time range
            base_timeframe: Base timeframe for resampling (default: 1m)
            **kwargs: Override any other fields

        """
        return cls(
            symbol=symbol,
            base_timeframe=base_timeframe,
            target_timeframes=target_timeframes,
            feature_config_version_info=feature_config,
            start_time=start_time,
            end_time=end_time,
            force_recompute=bool(kwargs.pop('force_recompute', True)),
            incremental_mode=bool(kwargs.pop('incremental_mode', True)),
            processing_context="inference",
            skip_existing_features=bool(kwargs.pop('skip_existing_features', False)),
            materialize_online=bool(kwargs.pop('materialize_online', True)),
            batch_size=int(str(kwargs.pop('batch_size'))) if 'batch_size' in kwargs and kwargs['batch_size'] is not None and isinstance(kwargs['batch_size'], (int, str)) else None,
            parallel_processing=bool(kwargs.pop('parallel_processing', True)),
            **kwargs
        )

    @classmethod
    def for_backfill(
        cls: type["FeaturePreprocessingRequest"],
        symbol: str,
        target_timeframes: List[Timeframe],
        feature_config: FeatureConfigVersionInfo,
        start_time: datetime,
        end_time: datetime,
        base_timeframe: Timeframe = Timeframe.MINUTE_1,
        batch_size: Optional[int] = 1000,
        **kwargs: object
    ) -> "FeaturePreprocessingRequest":
        """
        Create a preprocessing request for backfill operations.

        Backfill context defaults:
        - incremental_mode: False (full recomputation)
        - skip_existing_features: False (overwrite existing)
        - force_recompute: True (ensure complete historical data)
        - materialize_online: False (backfill is offline only)
        - batch_size: 1000 (chunked processing for large time ranges)
        - parallel_processing: True (maximize throughput)

        Args:
            symbol: Trading symbol (e.g., 'BTCUSD')
            target_timeframes: List of timeframes to compute features for
            feature_config: Feature configuration with version info
            start_time: Start of time range
            end_time: End of time range
            base_timeframe: Base timeframe for resampling (default: 1m)
            batch_size: Chunk size for processing (default: 1000)
            **kwargs: Override any other fields
        """
        return cls(
            symbol=symbol,
            base_timeframe=base_timeframe,
            target_timeframes=target_timeframes,
            feature_config_version_info=feature_config,
            start_time=start_time,
            end_time=end_time,
            force_recompute=bool(kwargs.pop('force_recompute', True)),
            incremental_mode=bool(kwargs.pop('incremental_mode', False)),
            processing_context="backfill",
            skip_existing_features=bool(kwargs.pop('skip_existing_features', False)),
            materialize_online=bool(kwargs.pop('materialize_online', False)),
            batch_size=batch_size if batch_size is not None else (int(str(kwargs.pop('batch_size'))) if 'batch_size' in kwargs and kwargs['batch_size'] is not None else None),
            parallel_processing=bool(kwargs.pop('parallel_processing', True)),
            **kwargs
        )

    # Utility Methods
    def get_enabled_features(self) -> List[FeatureDefinition]:
        """Get only enabled feature definitions as domain objects."""
        feature_definitions = []
        for feature_dict in self.feature_config_version_info.feature_definitions:
            if feature_dict.enabled:
                feature_def = FeatureDefinition(
                    name=feature_dict.name,
                    enabled=feature_dict.enabled,
                    derivatives=feature_dict.derivatives,
                    parameter_sets=feature_dict.parameter_sets,
                    parsed_parameter_sets={},
                )
                feature_definitions.append(feature_def)
        return feature_definitions

    def get_all_feature_definitions(self) -> List[FeatureDefinition]:
        """Get all feature definitions as domain objects (enabled and disabled)."""
        feature_definitions = []
        for feature_dict in self.feature_config_version_info.feature_definitions:
            feature_def = FeatureDefinition(
                name=feature_dict.name,
                enabled=feature_dict.enabled,
                derivatives=feature_dict.derivatives,
                parameter_sets=feature_dict.parameter_sets,
                parsed_parameter_sets={},
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
