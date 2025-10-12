"""
Test builder for FeaturePreprocessingRequest.

Provides a fluent API for creating test requests with sensible defaults.
This is a TEST-ONLY builder that wraps the production factory methods
with additional test-specific flexibility.
"""
from datetime import datetime, timezone
from typing import List, Optional

from drl_trading_common.config.feature_config import FeatureDefinition
from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo
from drl_trading_common.model.feature_preprocessing_request import FeaturePreprocessingRequest
from drl_trading_common.model.timeframe import Timeframe


class FeaturePreprocessingRequestBuilder:
    """
    Test-specific builder for FeaturePreprocessingRequest.

    Provides maximum flexibility for test scenarios while leveraging
    the production factory methods under the hood.

    Example usage:
        request = (FeaturePreprocessingRequestBuilder()
                   .for_inference()
                   .with_symbol("ETHUSD")
                   .with_target_timeframes([Timeframe.MINUTE_5, Timeframe.MINUTE_15])
                   .build())
    """

    def __init__(self) -> None:
        """Initialize builder with test defaults."""
        # Default feature definition for tests
        self._feature_definitions = [
            FeatureDefinition(
                name="test_feature",
                enabled=True,
                derivatives=[0],
                parameter_sets=[]
            )
        ]

        # Default version info for tests
        self._feature_config = FeatureConfigVersionInfo(
            semver="1.0.0",
            hash="test_hash",
            created_at=datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            feature_definitions=[fd.model_dump() for fd in self._feature_definitions],
            description="Test features"
        )

        # Request parameters
        self._symbol = "BTCUSD"
        self._base_timeframe = Timeframe.MINUTE_1
        self._target_timeframes = [Timeframe.MINUTE_5]
        self._start_time = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        self._end_time = datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        self._processing_context = "training"
        self._kwargs = {}  # Additional kwargs to pass through

    def with_symbol(self, symbol: str) -> 'FeaturePreprocessingRequestBuilder':
        """Set the symbol for the request."""
        self._symbol = symbol
        return self

    def with_base_timeframe(self, timeframe: Timeframe) -> 'FeaturePreprocessingRequestBuilder':
        """Set the base timeframe."""
        self._base_timeframe = timeframe
        return self

    def with_target_timeframes(self, timeframes: List[Timeframe]) -> 'FeaturePreprocessingRequestBuilder':
        """Set the target timeframes."""
        self._target_timeframes = timeframes
        return self

    def with_processing_context(self, context: str) -> 'FeaturePreprocessingRequestBuilder':
        """Set the processing context."""
        self._processing_context = context
        return self

    def for_training(self) -> 'FeaturePreprocessingRequestBuilder':
        """Configure for training context."""
        self._processing_context = "training"
        return self

    def for_inference(self) -> 'FeaturePreprocessingRequestBuilder':
        """Configure for inference context."""
        self._processing_context = "inference"
        return self

    def for_backfill(self) -> 'FeaturePreprocessingRequestBuilder':
        """Configure for backfill context."""
        self._processing_context = "backfill"
        return self

    def with_force_recompute(self, force: bool = True) -> 'FeaturePreprocessingRequestBuilder':
        """Enable/disable force recompute."""
        self._kwargs['force_recompute'] = force
        return self

    def with_skip_existing(self, skip: bool = True) -> 'FeaturePreprocessingRequestBuilder':
        """Enable/disable skipping existing features."""
        self._kwargs['skip_existing_features'] = skip
        return self

    def with_materialize_online(self, materialize: bool = True) -> 'FeaturePreprocessingRequestBuilder':
        """Enable/disable online materialization."""
        self._kwargs['materialize_online'] = materialize
        return self

    def with_request_id(self, request_id: str) -> 'FeaturePreprocessingRequestBuilder':
        """Set the request ID."""
        self._kwargs['request_id'] = request_id
        return self

    def with_time_range(
        self,
        start: datetime,
        end: datetime
    ) -> 'FeaturePreprocessingRequestBuilder':
        """Set the time range."""
        self._start_time = start
        self._end_time = end
        return self

    def with_feature_config(
        self,
        feature_config: FeatureConfigVersionInfo
    ) -> 'FeaturePreprocessingRequestBuilder':
        """Set custom feature configuration."""
        self._feature_config = feature_config
        return self

    def with_features(
        self,
        feature_definitions: List[FeatureDefinition]
    ) -> 'FeaturePreprocessingRequestBuilder':
        """Set custom feature definitions."""
        self._feature_definitions = feature_definitions
        # Update version info with new features
        self._feature_config = FeatureConfigVersionInfo(
            semver=self._feature_config.semver,
            hash=self._feature_config.hash,
            created_at=self._feature_config.created_at,
            feature_definitions=[fd.model_dump() for fd in feature_definitions],
            description=self._feature_config.description
        )
        return self

    def with_feature_names(self, *feature_names: str) -> 'FeaturePreprocessingRequestBuilder':
        """Set features by name (creates simple enabled features)."""
        feature_definitions = [
            FeatureDefinition(
                name=name,
                enabled=True,
                derivatives=[0],
                parameter_sets=[]
            )
            for name in feature_names
        ]
        return self.with_features(feature_definitions)

    def with_batch_size(self, batch_size: Optional[int]) -> 'FeaturePreprocessingRequestBuilder':
        """Set batch size for chunked processing."""
        self._kwargs['batch_size'] = batch_size
        return self

    def with_parallel_processing(self, parallel: bool = True) -> 'FeaturePreprocessingRequestBuilder':
        """Enable/disable parallel processing."""
        self._kwargs['parallel_processing'] = parallel
        return self

    def build(self) -> FeaturePreprocessingRequest:
        """
        Build the FeaturePreprocessingRequest.

        Uses the production factory methods based on context,
        ensuring test requests behave like production requests.
        """
        # Select appropriate factory method based on context
        if self._processing_context == "training":
            return FeaturePreprocessingRequest.for_training(
                symbol=self._symbol,
                target_timeframes=self._target_timeframes,
                feature_config=self._feature_config,
                start_time=self._start_time,
                end_time=self._end_time,
                base_timeframe=self._base_timeframe,
                **self._kwargs
            )
        elif self._processing_context == "inference":
            return FeaturePreprocessingRequest.for_inference(
                symbol=self._symbol,
                target_timeframes=self._target_timeframes,
                feature_config=self._feature_config,
                start_time=self._start_time,
                end_time=self._end_time,
                base_timeframe=self._base_timeframe,
                **self._kwargs
            )
        elif self._processing_context == "backfill":
            return FeaturePreprocessingRequest.for_backfill(
                symbol=self._symbol,
                target_timeframes=self._target_timeframes,
                feature_config=self._feature_config,
                start_time=self._start_time,
                end_time=self._end_time,
                base_timeframe=self._base_timeframe,
                **self._kwargs
            )
        else:
            # Fallback to direct construction for custom contexts
            return FeaturePreprocessingRequest(
                symbol=self._symbol,
                base_timeframe=self._base_timeframe,
                target_timeframes=self._target_timeframes,
                feature_config_version_info=self._feature_config,
                start_time=self._start_time,
                end_time=self._end_time,
                processing_context=self._processing_context,
                **self._kwargs
            )
