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
from typing import Any, Dict, List

import dask
import pandas as pd
from dask import delayed
from injector import inject
from pandas import DataFrame

from drl_trading_common.model.feature_preprocessing_request import (
    FeaturePreprocessingRequest,
)
from drl_trading_common.model.timeframe import Timeframe
from drl_trading_core.core.model.dataset_identifier import DatasetIdentifier
from drl_trading_core.core.model.feature_computation_request import (
    FeatureComputationRequest,
)
from drl_trading_core.core.model.feature_definition import FeatureDefinition
from drl_trading_preprocess.core.model.coverage.feature_coverage_analysis import (
    FeatureCoverageAnalysis,
    WarmupPeriod,
)
from drl_trading_preprocess.core.port.feature_store_save_port import (
    IFeatureStoreSavePort,
)
from drl_trading_preprocess.core.port.preprocessing_message_publisher_port import (
    PreprocessingMessagePublisherPort,
)
from drl_trading_preprocess.core.service.compute.computing_service import (
    FeatureComputingService,
)
from drl_trading_preprocess.core.service.coverage.feature_coverage_analyzer import (
    FeatureCoverageAnalyzer,
)
from drl_trading_preprocess.core.service.coverage.feature_coverage_evaluator import (
    FeatureCoverageEvaluator,
)
from drl_trading_preprocess.core.service.resample.market_data_resampling_service import (
    MarketDataResamplingService,
)
from drl_trading_preprocess.core.service.validate.feature_validator import (
    FeatureValidator,
)
from drl_trading_preprocess.infrastructure.config.preprocess_config import (
    DaskConfigs,
    FeatureComputationConfig,
)

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
        feature_coverage_evaluator: FeatureCoverageEvaluator,
        message_publisher: PreprocessingMessagePublisherPort,
        dask_configs: DaskConfigs,
        feature_computation_config: FeatureComputationConfig,
    ) -> None:
        """
        Initialize the preprocessing orchestrator with all required dependencies.

        Args:
            market_data_resampler: Service for resampling market data to higher timeframes
            feature_computer: Service for dynamic feature computation
            feature_validator: Service for validating feature definitions
            feature_store_port: Port for saving features to Feast
            feature_coverage_analyzer: Service for analyzing feature coverage
            feature_coverage_evaluator: Service for evaluating coverage analysis results
            message_publisher: Port for publishing preprocessing notifications
            dask_configs: Collection of Dask configurations for different workloads
            feature_computation_config: Configuration for feature computation (warmup settings)
        """
        self.market_data_resampler = market_data_resampler
        self.feature_computer = feature_computer
        self.feature_validator = feature_validator
        self.feature_store_port = feature_store_port
        self.feature_coverage_analyzer = feature_coverage_analyzer
        self.feature_coverage_evaluator = feature_coverage_evaluator
        self.message_publisher = message_publisher
        self.dask_configs = dask_configs
        self.feature_computation_config = feature_computation_config

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
        2. Feature coverage analysis (ONCE - reused by steps 3 and 5)
        3. Feature filtering based on skip_existing logic
        4. Market data resampling to target timeframes
        5. Feature warmup handling (reuses coverage from step 2)
        6. Dynamic feature computation
        7. Feature store persistence
        8. Async notification via Kafka

        Performance optimization:
        - Coverage analysis performed ONCE and reused by both skip_existing and warmup logic
        - Eliminates redundant Feast fetches and TimescaleDB queries
        - Dask parallelization for multi-timeframe analysis

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

            # Step 2: Analyze feature coverage (ONCE for entire pipeline)
            # This single analysis provides:
            # - features_needing_computation (for skip_existing logic)
            # - features_needing_warmup (for warmup phase)
            # - OHLCV availability constraints
            coverage_analyses = self._analyze_feature_coverage(request)

            # Step 3: Determine features to compute based on coverage
            features_to_compute = self._filter_features_to_compute(
                request, coverage_analyses
            )

            # Step 4: Early exit if no computation needed
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
            resampled_data = self._resample_market_data(request, coverage_analyses)

            # Step 5: Handle feature warmup if needed (reuses coverage_analyses and resampled_data)
            warmup_successful = self._handle_feature_warmup(
                request, features_to_compute, resampled_data, coverage_analyses
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
            # Subtract 2 for event_timestamp and symbol columns (metadata, not features)
            total_features = sum(len(features_df.columns) - 2 for features_df in computed_features.values())

            # Update performance tracking
            self._total_requests_processed += 1
            self._total_features_computed += total_features

            # Create success details
            success_details = {}
            for timeframe, features_df in computed_features.items():
                # Subtract 2 for event_timestamp and symbol columns (metadata, not features)
                success_details[f"features_{timeframe.value}"] = len(features_df.columns) - 2
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

    def _analyze_feature_coverage(
        self, request: FeaturePreprocessingRequest
    ) -> Dict[Timeframe, FeatureCoverageAnalysis]:
        """
        Analyze feature coverage for each target timeframe using Dask parallelization.

        This single analysis provides all coverage information needed by the entire pipeline:
        - features_needing_computation (for skip_existing optimization)
        - features_needing_warmup (for warmup phase)
        - OHLCV availability constraints
        - Existing feature data from Feast

        This method performs:
        1. OHLCV data availability check in TimescaleDB
        2. Batch fetch of existing features from Feast
        3. Coverage gap analysis and computation needs assessment

        Args:
            request: Feature computation request

        Returns:
            Dictionary mapping timeframes to coverage analysis results
        """
        feature_names = [f.name for f in request.get_enabled_features()]

        # Parallelize coverage analysis using Dask
        # Dask internally handles task queuing based on num_workers
        @delayed
        def analyze_timeframe(timeframe: Timeframe) -> tuple[Timeframe, FeatureCoverageAnalysis]:
            coverage_analysis = self.feature_coverage_analyzer.analyze_feature_coverage(
                symbol=request.symbol,
                timeframe=timeframe,
                base_timeframe=request.base_timeframe,
                feature_names=feature_names,
                requested_start_time=request.start_time,
                requested_end_time=request.end_time,
                feature_config_version_info=request.feature_config_version_info,
            )

            # Log cold start scenario
            if coverage_analysis.requires_resampling:
                logger.info(
                    f"Cold start detected for {request.symbol} {timeframe.value}: "
                    f"Target timeframe data will be resampled from {request.base_timeframe.value}"
                )

            logger.info(
                f"Feature coverage for {request.symbol} {timeframe.value}: "
                f"{len(self.feature_coverage_evaluator.get_fully_covered_features(coverage_analysis))} fully covered, "
                f"{len(self.feature_coverage_evaluator.get_partially_covered_features(coverage_analysis))} partial, "
                f"{len(self.feature_coverage_evaluator.get_missing_features(coverage_analysis))} missing"
            )
            return timeframe, coverage_analysis

        # Execute coverage analysis with Dask configuration
        # Dask will automatically queue tasks and execute them as workers become available
        total_timeframes = len(request.target_timeframes)

        logger.info(
            f"Analyzing feature coverage across {total_timeframes} timeframes "
            f"(scheduler={self.dask_configs.coverage_analysis.scheduler}, "
            f"num_workers={self.dask_configs.coverage_analysis.num_workers})"
        )

        # Create delayed tasks for all timeframes
        delayed_results = [analyze_timeframe(tf) for tf in request.target_timeframes]

        # Dask will handle parallelism based on scheduler and num_workers
        results = dask.compute(
            *delayed_results,
            scheduler=self.dask_configs.coverage_analysis.scheduler,
            num_workers=self.dask_configs.coverage_analysis.num_workers,
        )

        # Convert results to dictionary
        coverage_analyses = {timeframe: analysis for timeframe, analysis in results}
        return coverage_analyses

    def _filter_features_to_compute(
        self,
        request: FeaturePreprocessingRequest,
        coverage_analyses: Dict[Timeframe, FeatureCoverageAnalysis],
    ) -> List[FeatureDefinition]:
        """
        Determine which features need computation based on coverage analysis.

        When skip_existing_features=False or force_recompute=True:
        - Returns all enabled features (ignores coverage, computes everything)

        When skip_existing_features=True and force_recompute=False:
        - Returns only features that need computation based on coverage analysis

        Args:
            request: Original request
            coverage_analyses: Coverage analysis results per timeframe

        Returns:
            Filtered list of feature definitions to compute
        """
        # If force_recompute or not skipping existing, compute all enabled features
        if request.force_recompute or not request.skip_existing_features:
            logger.info("Computing all enabled features (skip_existing=False or force_recompute=True)")
            return request.get_enabled_features()

        # Otherwise, use coverage analysis to filter features
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
                needing_comp = self.feature_coverage_evaluator.get_features_needing_computation(analysis)
                if needing_comp:
                    features_needing_computation.update(needing_comp)
                    logger.debug(
                        f"Features needing computation for {timeframe.value} (fresh): {needing_comp}"
                    )
            else:
                # OHLCV data exists, check which features are missing
                needing_comp = self.feature_coverage_evaluator.get_features_needing_computation(analysis)
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
        self, request: FeaturePreprocessingRequest, coverage_analyses: Dict[Timeframe, FeatureCoverageAnalysis]
    ) -> Dict[Timeframe, DataFrame]:
        """
        Resample market data to all target timeframes.

        On cold start (first run), this resamples from base timeframe to create
        target timeframe data. On warm runs, this performs incremental updates
        for new data only.

        Behavior depends on request.processing_context:
        - "backfill": Stateless mode - uses full time range for reproducibility
        - "training"/"inference": Stateful mode - uses incremental processing

        Args:
            request: Feature computation request

        Returns:
            Dictionary mapping timeframes to resampled data
        """
        # Determine which timeframes actually need resampling
        target_timeframes = [
            analysis.timeframe
            for analysis in coverage_analyses.values()
            if analysis.requires_resampling
        ]

        if not target_timeframes:
            logger.info("No target timeframes require resampling. Skipping resampling step.")
            return {}

        logger.info(
            f"Resampling {request.symbol} from {request.base_timeframe.value} "
            f"to {[tf.value for tf in target_timeframes]} "
            f"(context: {request.processing_context})"
        )

        # Use the market data resampling service with processing context
        resampling_response = self.market_data_resampler.resample_symbol_data_incremental(
            symbol=request.symbol,
            base_timeframe=request.base_timeframe,
            target_timeframes=target_timeframes,
            processing_context=request.processing_context,
        )

        # Check if we got valid resampled data
        if not resampling_response.resampled_data:
            raise RuntimeError(f"Market data resampling failed: No data returned for {request.symbol}")

        # Convert resampled data to DataFrames for feature computation
        resampled_dataframes = {}
        for timeframe, market_data_list in resampling_response.resampled_data.items():
            if market_data_list:
                # Convert MarketDataModel list to DataFrame
                # Use capitalized column names for pandas_ta compatibility
                df_data = []
                for market_data in market_data_list:
                    df_data.append({
                        'timestamp': market_data.timestamp,
                        'Open': market_data.open_price,
                        'High': market_data.high_price,
                        'Low': market_data.low_price,
                        'Close': market_data.close_price,
                        'Volume': market_data.volume,
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
            computation_request = FeatureComputationRequest(
                dataset_id=DatasetIdentifier(
                    symbol=request.symbol,
                    timeframe=timeframe,
                ),
                feature_definitions=features_to_compute,
                market_data=market_data,
            )
            features_df = self.feature_computer.compute_batch(
                computation_request
            )

            if not features_df.empty:
                # Add required columns for feature store
                # event_timestamp: Required by Feast for temporal operations
                # symbol: Required as entity key for feature lookup
                features_df = features_df.copy()
                features_df['event_timestamp'] = pd.to_datetime(features_df.index)
                features_df['symbol'] = request.symbol

                computed_features[timeframe] = features_df
                logger.info(
                    f"Computed {len(features_df.columns) - 2} feature columns "  # -2 for event_timestamp and symbol
                    f"for {len(features_df)} data points on {timeframe.value}"
                )
            else:
                logger.warning(f"No features computed for {timeframe.value}")

        return computed_features

    def _handle_feature_warmup(
        self,
        request: FeaturePreprocessingRequest,
        features_to_compute: List[FeatureDefinition],
        resampled_data: Dict[Timeframe, DataFrame],
        coverage_analyses: Dict[Timeframe, FeatureCoverageAnalysis],
    ) -> bool:
        """
        Handle feature warmup using pre-computed coverage analysis and already-resampled data.

        This method REUSES the coverage analysis performed earlier in _analyze_feature_coverage()
        instead of re-analyzing, eliminating redundant database queries and Feast fetches.

        Warmup scenarios:
        1. Features fully covered by feast -> no warmup needed
        2. Features partially covered -> warmup with historical data up to coverage gap
        3. Features not in feast -> full warmup with ~500 OHLCV records

        This method reuses:
        - coverage_analyses: Pre-computed coverage from _analyze_feature_coverage()
        - resampled_data: Already resampled market data from _resample_market_data()

        Args:
            request: Feature computation request
            features_to_compute: Features that need computation
            resampled_data: Already resampled market data for all timeframes
            coverage_analyses: Pre-computed coverage analysis results (REUSED, not re-fetched)

        Returns:
            True if warmup successful or not needed, False on failure
        """
        logger.info(f"Handling feature warmup for {len(features_to_compute)} features")

        try:
            # Iterate through each timeframe and check if warmup is needed
            # REUSE the coverage analysis instead of calling feature_coverage_analyzer again
            for timeframe, coverage_analysis in coverage_analyses.items():
                # Check if warmup is needed for this timeframe
                if not self.feature_coverage_evaluator.get_features_needing_warmup(coverage_analysis):
                    logger.info(f"No warmup needed for {request.symbol} {timeframe.value}")
                    continue

                # Get warmup period from coverage analysis
                warmup_period = self.feature_coverage_evaluator.get_warmup_period(
                    coverage_analysis, warmup_candles=self.feature_computation_config.warmup_candles  # type: ignore[arg-type]
                )
                if not warmup_period:
                    logger.info(f"No warmup period calculated for {request.symbol} {timeframe.value}")
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
        warmup_period: WarmupPeriod,
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
        warmup_start, warmup_end = warmup_period.start_time, warmup_period.end_time

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
            warmup_computation_request = FeatureComputationRequest(
                dataset_id=DatasetIdentifier(
                    symbol=request.symbol,
                    timeframe=timeframe,
                ),
                feature_definitions=features_to_compute,
                market_data=warmup_df,
            )
            warmup_result = self.feature_computer.compute_batch(
                warmup_computation_request
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
                    feature_view_requests=[],
                    processing_context=request.processing_context,
                    requested_start_time=pd.to_datetime(request.start_time),
                    requested_end_time=pd.to_datetime(request.end_time),
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
        from drl_trading_core.core.model.dataset_identifier import DatasetIdentifier
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
