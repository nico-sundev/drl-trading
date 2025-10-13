"""
Preprocessing orchestrator that coordinates the complete feature computation pipeline.

This orchestrator is the heart of the drl-trading-preprocess package,
coordinating multiple specialized services to handle real-world scenarios including:
- Dynamic feature definitions per request
- Incremental processing with existing feature checking
- Multiple timeframe resampling
- Feast feature store integration
- Performance optimization for production deployment
- Parallel coverage analysis using Dask
"""
import logging
from datetime import datetime
from typing import Any, Dict, List

import dask
from dask import delayed
from injector import inject
from pandas import DataFrame

from drl_trading_common.config.feature_config import FeatureDefinition
from drl_trading_common.model.dataset_identifier import DatasetIdentifier
from drl_trading_common.model.feature_preprocessing_request import FeaturePreprocessingRequest
from drl_trading_common.model.timeframe import Timeframe
from drl_trading_preprocess.core.model.coverage.feature_coverage_analysis import (
    FeatureCoverageAnalysis,
)
from drl_trading_preprocess.core.port.feature_store_save_port import IFeatureStoreSavePort
from drl_trading_preprocess.core.port.preprocessing_message_publisher_port import PreprocessingMessagePublisherPort
from drl_trading_preprocess.core.service.compute.computing_service import FeatureComputingService
from drl_trading_preprocess.core.service.validate.feature_validator import FeatureValidator
from drl_trading_preprocess.core.service.resample.market_data_resampling_service import MarketDataResamplingService
from drl_trading_preprocess.core.service.coverage.feature_coverage_analyzer import FeatureCoverageAnalyzer

logger = logging.getLogger(__name__)


class PreprocessingOrchestrator:
    """
    Orchestrates the complete feature preprocessing pipeline.

    This orchestrator coordinates multiple specialized services to fulfill feature
    computation requests through an 8-step workflow:

    1. Request validation
    2. Existing feature coverage analysis
    3. Market data resampling to target timeframes
    4. Feature warmup for derivatives
    5. Feature computation across timeframes
    6. Feature store persistence
    7. Online materialization (optional)
    8. Async notification publishing

    Real-world capabilities:
    - Handles dynamic feature definitions per processing request
    - Prevents redundant computation by checking existing features
    - Supports incremental processing for continuous training
    - Manages multiple timeframe resampling efficiently
    - Integrates with Feast feature store for storage and retrieval
    - Optimized for production deployment with comprehensive error handling

    Architecture:
    - Follows Hexagonal Architecture (Ports & Adapters)
    - Delegates work to specialized services
    - Manages workflow coordination and error handling
    - Publishes fire-and-forget notifications via ports
    """

    @inject
    def __init__(
        self,
        market_data_resampler: MarketDataResamplingService,
        feature_computer: FeatureComputingService,
        feature_validator: FeatureValidator,
        feature_store_port: IFeatureStoreSavePort,
        feature_coverage_analyzer: FeatureCoverageAnalyzer,
        message_publisher: PreprocessingMessagePublisherPort,
    ) -> None:
        """
        Initialize the preprocessing orchestrator with all required dependencies.

        Args:
            market_data_resampler: Service for resampling market data to higher timeframes
            feature_computer: Service for dynamic feature computation
            feature_validator: Service for validating feature definitions
            feature_store_port: Port for saving features to Feast
            feature_coverage_analyzer: Service for analyzing feature coverage
            message_publisher: Port for publishing preprocessing notifications
        """
        self.market_data_resampler = market_data_resampler
        self.feature_computer = feature_computer
        self.feature_validator = feature_validator
        self.feature_store_port = feature_store_port
        self.feature_coverage_analyzer = feature_coverage_analyzer
        self.message_publisher = message_publisher

        # Performance tracking
        self._total_requests_processed = 0
        self._total_features_computed = 0
        self._total_processing_time_ms = 0

        logger.info("PreprocessingOrchestrator initialized with all dependencies")

    def process_feature_computation_request(
        self,
        request: FeaturePreprocessingRequest,
    ) -> None:
        """
        Process a complete feature computation request (fire-and-forget).

        This is the main orchestration entry point that coordinates the 8-step workflow:
        1. Request validation
        2. Existing feature checking (if enabled)
        3. Market data resampling to target timeframes
        4. Feature warmup handling
        5. Dynamic feature computation
        6. Feature store persistence
        7. Online materialization (optional)
        8. Async notification via Kafka

        Args:
            request: Complete feature computation request
        """

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

                # Publish notification that processing was skipped (all features already exist)
                self.message_publisher.publish_preprocessing_completed(
                    request=request,
                    processing_context=request.processing_context,
                    total_features_computed=0,
                    timeframes_processed=[],
                    success_details={"reason": "all_features_exist", "skipped": True}
                )
                return

            # Step 4: Resample market data to target timeframes (ONCE for all timeframes)
            resampled_data = self._resample_market_data(request)

            # Step 5: Handle feature warmup if needed (reuses resampled_data, no redundant resampling)
            warmup_successful = self._handle_feature_warmup(
                request, features_to_compute, resampled_data
            )
            if not warmup_successful:
                logger.error(f"Feature warmup failed for {request.symbol}")

                # Publish error notification for warmup failure
                self.message_publisher.publish_preprocessing_error(
                    request=request,
                    processing_context=request.processing_context,
                    error_message="Feature warmup failed",
                    error_details={"failed_step": "feature_warmup"},
                    failed_step="feature_warmup"
                )
                return

            # Step 6: Compute features for each timeframe
            computed_features = self._compute_features_for_timeframes(
                request, features_to_compute, resampled_data
            )

            # Step 7: Store features in feature store
            self._store_computed_features(request, computed_features)

            # Step 8: Publish successful completion notification
            total_features = sum(len(features_df.columns) for features_df in computed_features.values())

            # Update performance tracking
            self._total_requests_processed += 1
            self._total_features_computed += total_features

            # Create success details
            success_details = {}
            for timeframe, features_df in computed_features.items():
                success_details[f"features_{timeframe.value}"] = len(features_df.columns)
                success_details[f"records_{timeframe.value}"] = len(features_df)

            # Publish async notification (topic routing based on processing_context)
            self.message_publisher.publish_preprocessing_completed(
                request=request,
                processing_context=request.processing_context,
                total_features_computed=total_features,
                timeframes_processed=list(computed_features.keys()),
                success_details=success_details
            )

            logger.info(
                f"Successfully completed request {request.request_id}: "
                f"{total_features} features computed across {len(computed_features)} timeframes"
            )

        except Exception as e:

            # Publish error notification
            self.message_publisher.publish_preprocessing_error(
                request=request,
                processing_context=request.processing_context,
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
    ) -> Dict[Timeframe, FeatureCoverageAnalysis]:
        """
        Analyze feature coverage for each target timeframe using Dask parallelization.

        This method:
        1. Checks OHLCV data availability in TimescaleDB
        2. Batch fetches existing features from Feast
        3. Analyzes coverage gaps and computation needs

        Args:
            request: Feature computation request

        Returns:
            Dictionary mapping timeframes to coverage analysis results
        """
        feature_names = [f.name for f in request.get_enabled_features()]

        # Parallelize coverage analysis using Dask
        @delayed
        def analyze_timeframe(timeframe: Timeframe) -> tuple[Timeframe, FeatureCoverageAnalysis]:
            coverage_analysis = self.feature_coverage_analyzer.analyze_feature_coverage(
                symbol=request.symbol,
                timeframe=timeframe,
                feature_names=feature_names,
                requested_start_time=request.start_time,
                requested_end_time=request.end_time,
                feature_config_version_info=request.feature_config_version_info,
            )
            logger.info(
                f"Feature coverage for {request.symbol} {timeframe.value}: "
                f"{len(coverage_analysis.fully_covered_features)} fully covered, "
                f"{len(coverage_analysis.partially_covered_features)} partial, "
                f"{len(coverage_analysis.missing_features)} missing"
            )
            return timeframe, coverage_analysis

        # Execute coverage analysis in parallel
        logger.info(f"Analyzing feature coverage across {len(request.target_timeframes)} timeframes (parallel)")
        delayed_results = [analyze_timeframe(tf) for tf in request.target_timeframes]
        results = dask.compute(*delayed_results)

        # Convert to dictionary
        coverage_analyses = {timeframe: analysis for timeframe, analysis in results}
        return coverage_analyses

    def _filter_features_to_compute(
        self,
        request: FeaturePreprocessingRequest,
        coverage_analyses: Dict[Timeframe, FeatureCoverageAnalysis],
    ) -> List[FeatureDefinition]:
        """
        Determine which features need computation based on coverage analysis.

        Args:
            request: Original request
            coverage_analyses: Coverage analysis results per timeframe

        Returns:
            Filtered list of feature definitions to compute
        """
        if not coverage_analyses:
            return request.get_enabled_features()

        # Collect all features needing computation across all timeframes
        features_needing_computation = set()

        for timeframe, analysis in coverage_analyses.items():
            if not analysis.ohlcv_available:
                # No OHLCV data available for this timeframe
                # This means we need to resample from base timeframe first,
                # then compute all requested features on the resampled data
                logger.warning(
                    f"No OHLCV data available for {request.symbol} {timeframe.value}. "
                    f"Will resample from base timeframe and compute all features."
                )
                # Add all requested features for this timeframe
                needing_comp = analysis.features_needing_computation
                if needing_comp:
                    features_needing_computation.update(needing_comp)
                    logger.debug(
                        f"Features needing computation for {timeframe.value} (fresh): {needing_comp}"
                    )
            else:
                # OHLCV data exists, check which features are missing
                needing_comp = analysis.features_needing_computation
                if needing_comp:
                    features_needing_computation.update(needing_comp)
                    logger.info(
                        f"Features needing computation for {timeframe.value}: {needing_comp}"
                    )

        if not features_needing_computation:
            logger.info("All requested features are fully covered. No computation needed.")
            return []

        # Return feature definitions for features needing computation
        all_features = request.get_enabled_features()
        return [
            feature for feature in all_features
            if feature.name in features_needing_computation
        ]

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
        features_to_compute: List[FeatureDefinition],
        resampled_data: Dict[Timeframe, DataFrame]
    ) -> bool:
        """
        Handle feature warmup using coverage analysis and already-resampled data.

        Warmup scenarios:
        1. Features fully covered by feast -> no warmup needed
        2. Features partially covered -> warmup with historical data up to coverage gap
        3. Features not in feast -> full warmup with ~500 OHLCV records

        This method reuses the resampled data from _resample_market_data() to avoid
        redundant database calls and resampling operations.

        Coverage analysis is parallelized using Dask since it's read-only.

        Args:
            request: Feature computation request
            features_to_compute: Features that need computation
            resampled_data: Already resampled market data for all timeframes

        Returns:
            True if warmup successful or not needed, False on failure
        """
        logger.info(f"Handling feature warmup for {len(features_to_compute)} features")

        # Check if any features need warmup by analyzing coverage
        feature_names = [f.name for f in features_to_compute]

        try:
            # Parallelize warmup analysis using Dask
            @delayed
            def analyze_warmup_needs(timeframe: Timeframe) -> tuple[Timeframe, FeatureCoverageAnalysis, bool]:
                # Analyze coverage to determine warmup needs
                coverage_analysis = self.feature_coverage_analyzer.analyze_feature_coverage(
                    symbol=request.symbol,
                    timeframe=timeframe,
                    feature_names=feature_names,
                    requested_start_time=request.start_time,
                    requested_end_time=request.end_time,
                    feature_config_version_info=request.feature_config_version_info,
                )

                # Check if warmup is needed
                if not coverage_analysis.features_needing_warmup:
                    logger.info(f"No warmup needed for {request.symbol} {timeframe.value}")
                    return timeframe, coverage_analysis, False

                # Get warmup period from coverage analysis
                warmup_period = coverage_analysis.get_warmup_period(warmup_candles=500)

                if not warmup_period:
                    logger.info(f"No warmup period calculated for {request.symbol} {timeframe.value}")
                    return timeframe, coverage_analysis, False

                return timeframe, coverage_analysis, True

            # Execute warmup analysis in parallel
            logger.info(f"Analyzing warmup needs across {len(request.target_timeframes)} timeframes (parallel)")
            delayed_analyses = [analyze_warmup_needs(tf) for tf in request.target_timeframes]
            warmup_analyses = dask.compute(*delayed_analyses)

            # Perform actual warmup for timeframes that need it
            for timeframe, coverage_analysis, needs_warmup in warmup_analyses:
                if not needs_warmup:
                    continue

                warmup_period = coverage_analysis.get_warmup_period(warmup_candles=500)
                if not warmup_period:
                    continue

                # Perform warmup using ALREADY RESAMPLED DATA
                success = self._perform_feature_warmup(
                    request, features_to_compute, timeframe, warmup_period, resampled_data
                )

                if not success:
                    logger.error(f"Failed to warmup features for {timeframe.value}")
                    return False

            logger.info("Feature warmup completed successfully")
            return True

        except Exception as e:
            logger.error(f"Feature warmup failed: {e}")
            return False

    def _perform_feature_warmup(
        self,
        request: FeaturePreprocessingRequest,
        features_to_compute: List[FeatureDefinition],
        timeframe: Timeframe,
        warmup_period: tuple[datetime, datetime],
        resampled_data: Dict[Timeframe, DataFrame]
    ) -> bool:
        """
        Perform actual feature warmup using already-resampled data.

        This method reuses the resampled data from _resample_market_data() instead of
        calling the resampling service again, avoiding redundant database queries and
        resampling computations.

        Args:
            request: Feature computation request
            features_to_compute: Features to warmup
            timeframe: Target timeframe
            warmup_period: Tuple of (start_time, end_time) for warmup
            resampled_data: Already resampled market data for all timeframes

        Returns:
            True if warmup successful, False otherwise
        """
        warmup_start, warmup_end = warmup_period

        logger.info(
            f"Performing feature warmup for {request.symbol} {timeframe.value}: "
            f"[{warmup_start} - {warmup_end}]"
        )

        try:
            # Use already resampled data instead of calling resample_symbol_data_incremental again
            if timeframe not in resampled_data:
                logger.warning(f"No resampled data available for {request.symbol} {timeframe.value}")
                return False

            warmup_df = resampled_data[timeframe]

            # Filter to warmup period
            warmup_df = warmup_df[
                (warmup_df.index >= warmup_start) &
                (warmup_df.index < warmup_end)
            ]

            if warmup_df.empty:
                logger.warning(
                    f"No warmup data in specified period for {request.symbol} {timeframe.value}"
                )
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
                # Use the feature version info from the request
                self.feature_store_port.store_computed_features_offline(
                    features_df=features_df,
                    symbol=request.symbol,
                    feature_version_info=request.feature_config_version_info,
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
