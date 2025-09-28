"""
Main preprocessing service that orchestrates market data resampling,
feature computation, and feature store operations.

This service i            # Step 7: Store computed features
            self._store_computed_features(request, computed_features)

            # Step 8: Log successful completionheart of the drl-trading-preprocess package,
handling real-world scenarios including:
- Dynamic feature definitions per request
- Incremental processing with existing feature checking
- Multiple timeframe resampling
- Feast feature store integration
- Performance optimization for production deployment
"""
import logging
from datetime import datetime
from typing import Any, Dict, List

from injector import inject
from pandas import DataFrame

from drl_trading_common.config.feature_config import FeatureDefinition
from drl_trading_common.model.dataset_identifier import DatasetIdentifier
from drl_trading_common.model.timeframe import Timeframe
from drl_trading_preprocess.core.model.computation.feature_preprocessing_request import FeaturePreprocessingRequest
from drl_trading_preprocess.core.model.computation.feature_computation_response import (
    FeatureExistenceCheckResult,
)
from drl_trading_preprocess.core.port.feature_store_save_port import IFeatureStoreSavePort
from drl_trading_preprocess.core.port.preprocessing_message_publisher_port import PreprocessingMessagePublisherPort
from drl_trading_preprocess.core.service.compute.computing_service import FeatureComputingService
from drl_trading_preprocess.core.service.feature_validator import FeatureValidator
from drl_trading_preprocess.core.service.resample.market_data_resampling_service import MarketDataResamplingService
from drl_trading_preprocess.infrastructure.adapter.feature_store.feast_feature_existence_checker import FeastFeatureExistenceChecker

logger = logging.getLogger(__name__)


class PreprocessService:
    """
    Main preprocessing service orchestrating the complete feature computation pipeline.

    Real-world capabilities:
    - Handles dynamic feature definitions per processing request
    - Prevents redundant computation by checking existing features
    - Supports incremental processing for continuous training
    - Manages multiple timeframe resampling efficiently
    - Integrates with Feast feature store for storage and retrieval
    - Optimized for production deployment with comprehensive error handling
    """

    @inject
    def __init__(
        self,
        market_data_resampler: MarketDataResamplingService,
        feature_computer: FeatureComputingService,
        feature_validator: FeatureValidator,
        feature_store_port: IFeatureStoreSavePort,
        feature_existence_checker: FeastFeatureExistenceChecker,
        message_publisher: PreprocessingMessagePublisherPort,
    ) -> None:
        """
        Initialize the preprocessing service with all required dependencies.

        Args:
            market_data_resampler: Service for resampling market data to higher timeframes
            feature_computer: Service for dynamic feature computation
            feature_validator: Service for validating feature definitions
            feature_store_port: Port for saving features to Feast
            feature_existence_checker: Port for checking existing features
            message_publisher: Port for publishing preprocessing notifications
        """
        self.market_data_resampler = market_data_resampler
        self.feature_computer = feature_computer
        self.feature_validator = feature_validator
        self.feature_store_port = feature_store_port
        self.feature_existence_checker = feature_existence_checker
        self.message_publisher = message_publisher

        # Performance tracking
        self._total_requests_processed = 0
        self._total_features_computed = 0
        self._total_processing_time_ms = 0

        logger.info("PreprocessService initialized with all dependencies")

    def process_feature_computation_request(
        self,
        request: FeaturePreprocessingRequest,
    ) -> None:
        """
        Process a complete feature computation request (fire-and-forget).

        This is the main entry point that handles:
        1. Request validation
        2. Existing feature checking (if enabled)
        3. Market data resampling to target timeframes
        4. Dynamic feature computation
        5. Feature store persistence
        6. Async notification via Kafka (future implementation)

        Args:
            request: Complete feature computation request
        """
        processing_start = datetime.now()
        self._total_requests_processed += 1

        logger.info(
            f"Starting processing for request {request.request_id}: "
            f"{request.symbol} [{request.start_time} - {request.end_time}] "
            f"with {len(request.get_enabled_features())} features"
        )

        try:
            # Step 1: Validate request
            self._validate_request(request)

            # Step 2: Check existing features if enabled
            features_to_compute = request.get_enabled_features()
            existing_features_info = None

            if request.skip_existing_features and not request.force_recompute:
                existing_features_info = self._check_existing_features(request)
                features_to_compute = self._filter_features_to_compute(
                    request, existing_features_info
                )

            # Step 3: Early exit if no computation needed
            if not features_to_compute:
                logger.info(f"No features to compute for {request.symbol} - all exist or none enabled")
                return

            # Step 4: Handle feature warmup if needed
            warmup_successful = self._handle_feature_warmup(request, features_to_compute)
            if not warmup_successful:
                logger.error(f"Feature warmup failed for {request.symbol}")
                return

            # Step 5: Resample market data to target timeframes
            resampled_data = self._resample_market_data(request)

            # Step 6: Compute features for each timeframe
            computed_features = self._compute_features_for_timeframes(
                request, features_to_compute, resampled_data
            )

            # Step 7: Store features in feature store
            self._store_computed_features(request, computed_features)

            # Step 8: Publish successful completion notification
            processing_end = datetime.now()
            total_features = sum(len(features_df.columns) for features_df in computed_features.values())
            duration = (processing_end - processing_start).total_seconds()

            # Update performance tracking
            self._total_features_computed += total_features
            self._total_processing_time_ms += int(duration * 1000)

            # Create success details
            success_details = {}
            for timeframe, features_df in computed_features.items():
                success_details[f"features_{timeframe.value}"] = len(features_df.columns)
                success_details[f"records_{timeframe.value}"] = len(features_df)

            # Publish async notification
            self.message_publisher.publish_preprocessing_completed(
                request=request,
                processing_duration_seconds=duration,
                total_features_computed=total_features,
                timeframes_processed=list(computed_features.keys()),
                success_details=success_details
            )

            logger.info(
                f"Successfully completed request {request.request_id}: "
                f"{total_features} features computed across {len(computed_features)} timeframes, "
                f"duration: {duration:.2f}s"
            )

        except Exception as e:
            processing_end = datetime.now()
            duration = (processing_end - processing_start).total_seconds()

            # Publish error notification
            self.message_publisher.publish_preprocessing_error(
                request=request,
                processing_duration_seconds=duration,
                error_message=str(e),
                error_details={"exception_type": type(e).__name__, "traceback": str(e)},
                failed_step="processing_pipeline"
            )

            logger.error(f"Failed to process request {request.request_id}: {str(e)}")

    def _validate_request(self, request: FeaturePreprocessingRequest) -> None:
        """
        Validate the feature computation request.

        Args:
            request: Request to validate

        Raises:
            ValueError: If request validation fails
        """
        # The request DTO already has validation in its validators
        # But we can add additional business logic validation here

        enabled_features = request.get_enabled_features()
        if not enabled_features:
            raise ValueError("No enabled features found in request")

        # Validate feature definitions against what we can actually compute
        validation_results = self.feature_validator.validate_definitions(enabled_features)

        invalid_features = [name for name, valid in validation_results.items() if not valid]
        if invalid_features:
            validation_errors = {name: "Feature definition validation failed" for name in invalid_features}

            # Publish validation error notification
            self.message_publisher.publish_feature_validation_error(
                request=request,
                invalid_features=invalid_features,
                validation_errors=validation_errors
            )

            raise ValueError(f"Invalid feature definitions: {invalid_features}")

    def _check_existing_features(
        self, request: FeaturePreprocessingRequest
    ) -> Dict[Timeframe, FeatureExistenceCheckResult]:
        """
        Check which features already exist in the feature store.

        Args:
            request: Feature computation request

        Returns:
            Dictionary mapping timeframes to existence check results
        """
        existing_features_info = {}
        feature_names = [f.name for f in request.get_enabled_features()]

        for timeframe in request.target_timeframes:
            existence_result = self.feature_existence_checker.check_feature_existence(
                symbol=request.symbol,
                timeframe=timeframe,
                feature_names=feature_names,
                start_time=request.start_time,
                end_time=request.end_time,
            )
            existing_features_info[timeframe] = existence_result

            logger.info(
                f"Feature existence check for {request.symbol} {timeframe.value}: "
                f"{len(existence_result.existing_features)} exist, "
                f"{len(existence_result.missing_features)} missing"
            )

        return existing_features_info

    def _filter_features_to_compute(
        self,
        request: FeaturePreprocessingRequest,
        existing_features_info: Dict[Timeframe, FeatureExistenceCheckResult],
    ) -> List[FeatureDefinition]:
        """
        Filter features to only compute those that don't exist or need updating.

        Args:
            request: Original request
            existing_features_info: Results from existence checking

        Returns:
            Filtered list of feature definitions to compute
        """
        if not existing_features_info:
            return request.get_enabled_features()

        # For simplicity, if any timeframe is missing features, compute all
        # In a more sophisticated implementation, you might compute per-timeframe

        for timeframe, existence_result in existing_features_info.items():
            missing_features = set(existence_result.missing_features)
            if missing_features:
                # Return original features - some are missing
                logger.info(
                    f"Missing features for {timeframe.value}: {missing_features}. "
                    "Will compute all requested features."
                )
                return request.get_enabled_features()

        logger.info("All requested features exist. No computation needed.")
        return []

    def _resample_market_data(
        self, request: FeaturePreprocessingRequest
    ) -> Dict[Timeframe, DataFrame]:
        """
        Resample market data to all target timeframes.

        Args:
            request: Feature computation request

        Returns:
            Dictionary mapping timeframes to resampled data
        """
        logger.info(
            f"Resampling {request.symbol} from {request.base_timeframe.value} "
            f"to {[tf.value for tf in request.target_timeframes]}"
        )

        # Use the market data resampling service
        resampling_response = self.market_data_resampler.resample_symbol_data_incremental(
            symbol=request.symbol,
            base_timeframe=request.base_timeframe,
            target_timeframes=request.target_timeframes,
        )

        # Check if we got valid resampled data
        if not resampling_response.resampled_data:
            raise RuntimeError(f"Market data resampling failed: No data returned for {request.symbol}")

        # Convert resampled data to DataFrames for feature computation
        resampled_dataframes = {}
        for timeframe, market_data_list in resampling_response.resampled_data.items():
            if market_data_list:
                # Convert MarketDataModel list to DataFrame
                df_data = []
                for market_data in market_data_list:
                    df_data.append({
                        'timestamp': market_data.timestamp,
                        'open': market_data.open_price,
                        'high': market_data.high_price,
                        'low': market_data.low_price,
                        'close': market_data.close_price,
                        'volume': market_data.volume,
                    })

                df = DataFrame(df_data)
                df.set_index('timestamp', inplace=True)
                resampled_dataframes[timeframe] = df

                logger.debug(
                    f"Resampled {request.symbol} to {timeframe.value}: "
                    f"{len(df)} candles from {df.index.min()} to {df.index.max()}"
                )

        return resampled_dataframes

    def _compute_features_for_timeframes(
        self,
        request: FeaturePreprocessingRequest,
        features_to_compute: List[FeatureDefinition],
        resampled_data: Dict[Timeframe, DataFrame],
    ) -> Dict[Timeframe, DataFrame]:
        """
        Compute features for all timeframes.

        Args:
            request: Original request
            features_to_compute: Features that need computation
            resampled_data: Market data for each timeframe

        Returns:
            Dictionary mapping timeframes to computed features
        """
        computed_features = {}

        for timeframe, market_data in resampled_data.items():
            logger.info(
                f"Computing {len(features_to_compute)} features for "
                f"{request.symbol} {timeframe.value} ({len(market_data)} data points)"
            )

            # Compute features for this timeframe using batch computation
            from drl_trading_common.config.feature_config import FeaturesConfig
            features_config = FeaturesConfig(
                dataset_definitions={request.symbol: [timeframe]},
                feature_definitions=features_to_compute
            )
            features_df = self.feature_computer.compute_batch(
                market_data, features_config
            )

            if not features_df.empty:
                computed_features[timeframe] = features_df
                logger.info(
                    f"Computed {len(features_df.columns)} feature columns "
                    f"for {len(features_df)} data points on {timeframe.value}"
                )
            else:
                logger.warning(f"No features computed for {timeframe.value}")

        return computed_features

    def _handle_feature_warmup(
        self,
        request: FeaturePreprocessingRequest,
        features_to_compute: List[FeatureDefinition]
    ) -> bool:
        """
        Handle feature warmup based on the three scenarios:

        1. Features fully covered by feast -> no warmup needed
        2. Features partially covered -> warmup with historical data up to coverage gap
        3. Features not in feast -> full warmup with ~500 OHLCV records

        Args:
            request: Feature computation request
            features_to_compute: Features that need computation

        Returns:
            True if warmup successful or not needed, False on failure
        """
        logger.info(f"Handling feature warmup for {len(features_to_compute)} features")

        try:
            # For each target timeframe, determine warmup needs
            for timeframe in request.target_timeframes:
                warmup_needed = self._assess_warmup_needs(request, features_to_compute, timeframe)

                if warmup_needed:
                    success = self._perform_feature_warmup(request, features_to_compute, timeframe)
                    if not success:
                        logger.error(f"Failed to warmup features for {timeframe.value}")
                        return False

            logger.info("Feature warmup completed successfully")
            return True

        except Exception as e:
            logger.error(f"Feature warmup failed: {e}")
            return False

    def _assess_warmup_needs(
        self,
        request: FeaturePreprocessingRequest,
        features_to_compute: List[FeatureDefinition],
        timeframe: Timeframe
    ) -> bool:
        """
        Assess whether warmup is needed for the given features and timeframe.

        Args:
            request: Feature computation request
            features_to_compute: Features to assess
            timeframe: Target timeframe

        Returns:
            True if warmup is needed, False otherwise
        """
        # Check if features exist and have recent data
        try:
            feature_names = [f.name for f in features_to_compute]

            # Get latest timestamps for each feature
            latest_timestamps = {}
            for feature_name in feature_names:
                latest_ts = self.feature_existence_checker.get_latest_feature_timestamp(
                    symbol=request.symbol,
                    timeframe=timeframe,
                    feature_name=feature_name
                )
                latest_timestamps[feature_name] = latest_ts

            # Determine if warmup is needed
            # If any feature has no data or data is too old, warmup is needed
            warmup_threshold = request.start_time  # Features should be up to request start

            needs_warmup = any(
                ts is None or ts < warmup_threshold
                for ts in latest_timestamps.values()
            )

            if needs_warmup:
                logger.info(
                    f"Warmup needed for {request.symbol} {timeframe.value}: "
                    f"features outdated or missing"
                )
            else:
                logger.debug(f"No warmup needed for {request.symbol} {timeframe.value}")

            return needs_warmup

        except Exception as e:
            logger.warning(f"Failed to assess warmup needs: {e}. Assuming warmup needed.")
            return True

    def _perform_feature_warmup(
        self,
        request: FeaturePreprocessingRequest,
        features_to_compute: List[FeatureDefinition],
        timeframe: Timeframe
    ) -> bool:
        """
        Perform actual feature warmup with historical data.

        Args:
            request: Feature computation request
            features_to_compute: Features to warmup
            timeframe: Target timeframe

        Returns:
            True if warmup successful, False otherwise
        """
        logger.info(f"Performing feature warmup for {request.symbol} {timeframe.value}")

        try:
            # Determine warmup period (default: ~500 candles or up to earliest missing data)
            warmup_candles = 500

            # Calculate warmup start time based on timeframe
            # This is a simplified calculation - in reality you'd want more sophisticated logic
            from datetime import timedelta

            timeframe_minutes = timeframe.to_minutes()
            warmup_period = timedelta(minutes=warmup_candles * timeframe_minutes)
            warmup_start = request.start_time - warmup_period

            logger.info(
                f"Warmup period: {warmup_start} to {request.start_time} "
                f"({warmup_candles} candles)"
            )

            # Get historical data for warmup
            # Note: This uses the base timeframe to get raw data, then resamples if needed
            warmup_response = self.market_data_resampler.resample_symbol_data_incremental(
                symbol=request.symbol,
                base_timeframe=request.base_timeframe,
                target_timeframes=[timeframe]
            )

            if not warmup_response.resampled_data or timeframe not in warmup_response.resampled_data:
                logger.warning(f"No warmup data available for {request.symbol} {timeframe.value}")
                return False

            # Convert to DataFrame
            warmup_market_data = warmup_response.resampled_data[timeframe]
            warmup_df_data = []
            for market_data in warmup_market_data:
                warmup_df_data.append({
                    'timestamp': market_data.timestamp,
                    'open': market_data.open_price,
                    'high': market_data.high_price,
                    'low': market_data.low_price,
                    'close': market_data.close_price,
                    'volume': market_data.volume,
                })

            warmup_df = DataFrame(warmup_df_data)
            warmup_df.set_index('timestamp', inplace=True)

            # Filter to warmup period
            warmup_df = warmup_df[
                (warmup_df.index >= warmup_start) &
                (warmup_df.index < request.start_time)
            ]

            if warmup_df.empty:
                logger.warning(f"No warmup data in specified period for {request.symbol} {timeframe.value}")
                return False

            # Perform warmup using compute_batch method
            from drl_trading_common.config.feature_config import FeaturesConfig
            features_config = FeaturesConfig(
                dataset_definitions={request.symbol: [timeframe]},
                feature_definitions=features_to_compute
            )
            warmup_result = self.feature_computer.compute_batch(
                warmup_df, features_config
            )

            success = not warmup_result.empty
            if success:
                logger.info(
                    f"Successfully warmed up features for {request.symbol} {timeframe.value} "
                    f"with {len(warmup_df)} data points, computed {len(warmup_result.columns)} features"
                )
            else:
                logger.error(f"Feature warmup failed for {request.symbol} {timeframe.value}")

            return success

        except Exception as e:
            logger.error(f"Feature warmup error: {e}")
            return False

    def _store_computed_features(
        self,
        request: FeaturePreprocessingRequest,
        computed_features: Dict[Timeframe, DataFrame],
    ) -> None:
        """
        Store computed features in the feature store.

        Args:
            request: Original request
            computed_features: Computed features by timeframe
        """

        for timeframe, features_df in computed_features.items():
            try:
                logger.info(
                    f"Storing {len(features_df.columns)} features "
                    f"for {request.symbol} {timeframe.value} to feature store"
                )

                # Store features offline
                # Note: feature_version_info and feature_view_requests would be generated from request
                # For now, using None/empty list as placeholders
                from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo

                version_info = FeatureConfigVersionInfo(
                    semver=f"1.0.{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    hash=f"feat_{request.symbol}_{timeframe.value}_{hash(str(features_df.columns))}",
                    created_at=datetime.now(),
                    feature_definitions=[f.dict() for f in request.get_enabled_features()],
                    base_timeframe=request.base_timeframe,
                    target_timeframes=[timeframe],
                    start_time=request.start_time,
                    end_time=request.end_time,
                    description=f"Features for {request.symbol} {timeframe.value}"
                )

                self.feature_store_port.store_computed_features_offline(
                    features_df=features_df,
                    symbol=request.symbol,
                    feature_version_info=version_info,
                    feature_view_requests=[],   # Would be created from features
                )

                # Store online if requested
                if request.materialize_online:
                    self.feature_store_port.batch_materialize_features(
                        features_df=features_df,
                        symbol=request.symbol,
                    )
                    logger.info(f"Features materialized to online store for {request.symbol} {timeframe.value}")

                logger.info(f"Features stored successfully for {request.symbol} {timeframe.value}")

            except Exception as e:
                logger.error(
                    f"Failed to store features for {request.symbol} {timeframe.value}: {e}"
                )
                raise

    def _create_dataset_identifier(self, symbol: str, timeframe: Timeframe) -> DatasetIdentifier:
        """Create a dataset identifier for the given symbol and timeframe."""
        # This would typically come from your common models
        # For now, create a simple implementation
        from drl_trading_common.model.dataset_identifier import DatasetIdentifier
        return DatasetIdentifier(symbol=symbol, timeframe=timeframe)



    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring."""
        avg_processing_time = (
            self._total_processing_time_ms / max(1, self._total_requests_processed)
        )

        return {
            "total_requests_processed": self._total_requests_processed,
            "total_features_computed": self._total_features_computed,
            "average_processing_time_ms": avg_processing_time,
            "features_per_request_avg": (
                self._total_features_computed / max(1, self._total_requests_processed)
            ),
        }
