"""
Main preprocessing service that orchestrates market data resampling,
feature computation, and feature store operations.

This service is the heart of the drl-trading-preprocess package,
handling real-world scenarios including:
- Dynamic feature definitions per request
- Incremental processing with existing feature checking
- Multiple timeframe resampling
- Feast feature store integration
- Performance optimization for production deployment
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from injector import inject
from pandas import DataFrame

from drl_trading_common.config.feature_config import FeatureDefinition
from drl_trading_common.model.dataset_identifier import DatasetIdentifier
from drl_trading_common.model.timeframe import Timeframe
from drl_trading_preprocess.core.model.feature_computation_request import FeatureComputationRequest
from drl_trading_preprocess.core.model.feature_computation_response import (
    FeatureComputationResponse,
    FeatureProcessingStats,
    FeatureStoreMetadata,
    FeatureExistenceCheckResult,
)
from drl_trading_preprocess.core.port.feature_existence_check_port import IFeatureExistenceCheckPort
from drl_trading_preprocess.core.port.feature_store_save_port import IFeatureStoreSavePort
from drl_trading_preprocess.core.service.dynamic_feature_computing_service import IDynamicFeatureComputer
from drl_trading_preprocess.core.service.market_data_resampling_service import MarketDataResamplingService

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
        feature_computer: IDynamicFeatureComputer,
        feature_store_port: IFeatureStoreSavePort,
        feature_existence_checker: IFeatureExistenceCheckPort,
    ) -> None:
        """
        Initialize the preprocessing service with all required dependencies.

        Args:
            market_data_resampler: Service for resampling market data to higher timeframes
            feature_computer: Service for dynamic feature computation
            feature_store_port: Port for saving features to Feast
            feature_existence_checker: Port for checking existing features
        """
        self.market_data_resampler = market_data_resampler
        self.feature_computer = feature_computer
        self.feature_store_port = feature_store_port
        self.feature_existence_checker = feature_existence_checker

        # Performance tracking
        self._total_requests_processed = 0
        self._total_features_computed = 0
        self._total_processing_time_ms = 0

        logger.info("PreprocessService initialized with all dependencies")

    def process_feature_computation_request(
        self,
        request: FeatureComputationRequest,
    ) -> FeatureComputationResponse:
        """
        Process a complete feature computation request.

        This is the main entry point that handles:
        1. Request validation
        2. Existing feature checking (if enabled)
        3. Market data resampling to target timeframes
        4. Dynamic feature computation
        5. Feature store persistence
        6. Comprehensive response generation

        Args:
            request: Complete feature computation request

        Returns:
            Detailed response with processing results and metadata
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
                return self._create_no_computation_response(
                    request, processing_start, existing_features_info
                )

            # Step 4: Resample market data to target timeframes
            resampled_data = self._resample_market_data(request)

            # Step 5: Compute features for each timeframe
            computed_features = self._compute_features_for_timeframes(
                request, features_to_compute, resampled_data
            )

            # Step 6: Store features in feature store
            feature_store_metadata = self._store_computed_features(
                request, computed_features
            )

            # Step 7: Create success response
            processing_end = datetime.now()
            response = self._create_success_response(
                request,
                processing_start,
                processing_end,
                computed_features,
                feature_store_metadata,
                existing_features_info,
            )

            self._update_performance_metrics(response)

            logger.info(
                f"Successfully completed request {request.request_id}: "
                f"{response.stats.features_computed} features computed, "
                f"{response.stats.features_skipped} skipped, "
                f"duration: {response.get_processing_duration_seconds():.2f}s"
            )

            return response

        except Exception as e:
            logger.error(f"Failed to process request {request.request_id}: {str(e)}")
            return self._create_error_response(request, processing_start, str(e))

    def _validate_request(self, request: FeatureComputationRequest) -> None:
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
        validation_results = self.feature_computer.validate_feature_definitions(
            enabled_features,
            # Create a dataset identifier for validation
            # We'll use the first target timeframe for validation
            dataset_id=self._create_dataset_identifier(request.symbol, request.target_timeframes[0]),
        )

        invalid_features = [name for name, valid in validation_results.items() if not valid]
        if invalid_features:
            raise ValueError(f"Invalid feature definitions: {invalid_features}")

    def _check_existing_features(
        self, request: FeatureComputationRequest
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
        request: FeatureComputationRequest,
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
        self, request: FeatureComputationRequest
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
        request: FeatureComputationRequest,
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

            # Create a modified request for this specific timeframe
            timeframe_request = FeatureComputationRequest(
                symbol=request.symbol,
                base_timeframe=request.base_timeframe,
                target_timeframes=[timeframe],  # Single timeframe
                feature_definitions=features_to_compute,
                start_time=request.start_time,
                end_time=request.end_time,
                request_id=f"{request.request_id}_{timeframe.value}",
                force_recompute=request.force_recompute,
                incremental_mode=request.incremental_mode,
                processing_context=request.processing_context,
            )

            # Compute features for this timeframe
            features_df = self.feature_computer.compute_features_for_request(
                timeframe_request, market_data
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

    def _store_computed_features(
        self,
        request: FeatureComputationRequest,
        computed_features: Dict[Timeframe, DataFrame],
    ) -> FeatureStoreMetadata:
        """
        Store computed features in the feature store.

        Args:
            request: Original request
            computed_features: Computed features by timeframe

        Returns:
            Metadata about feature store operations
        """
        metadata = FeatureStoreMetadata()

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
                    version=f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    creation_timestamp=datetime.now(),
                    feature_count=len(features_df.columns),
                    dataset_identifier=self._create_dataset_identifier(request.symbol, timeframe)
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
                    metadata.online_store_updated = True

                metadata.offline_store_paths.append(f"{request.symbol}_{timeframe.value}")

            except Exception as e:
                logger.error(
                    f"Failed to store features for {request.symbol} {timeframe.value}: {e}"
                )
                raise

        metadata.materialization_completed = True
        metadata.computation_timestamp = datetime.now()

        return metadata

    def _create_dataset_identifier(self, symbol: str, timeframe: Timeframe) -> DatasetIdentifier:
        """Create a dataset identifier for the given symbol and timeframe."""
        # This would typically come from your common models
        # For now, create a simple implementation
        from drl_trading_common.model.dataset_identifier import DatasetIdentifier
        return DatasetIdentifier(symbol=symbol, timeframe=timeframe)

    def _create_success_response(
        self,
        request: FeatureComputationRequest,
        processing_start: datetime,
        processing_end: datetime,
        computed_features: Dict[Timeframe, DataFrame],
        feature_store_metadata: FeatureStoreMetadata,
        existing_features_info: Optional[Dict[Timeframe, FeatureExistenceCheckResult]],
    ) -> FeatureComputationResponse:
        """Create a successful response."""

        # Calculate statistics
        total_features_computed = sum(len(df.columns) for df in computed_features.values())
        total_data_points = sum(len(df) for df in computed_features.values())

        features_skipped = 0
        if existing_features_info:
            for existence_result in existing_features_info.values():
                features_skipped += len(existence_result.existing_features)

        processing_duration_ms = int((processing_end - processing_start).total_seconds() * 1000)

        stats = FeatureProcessingStats(
            features_requested=len(request.get_enabled_features()) * len(request.target_timeframes),
            features_computed=total_features_computed,
            features_skipped=features_skipped,
            features_failed=0,  # No failures if we got here
            timeframes_processed={tf.value: len(df) for tf, df in computed_features.items()},
            processing_duration_ms=processing_duration_ms,
            data_points_processed=total_data_points,
            computation_rate_per_second=(
                total_data_points / max(0.001, processing_duration_ms / 1000.0)
            ),
        )

        return FeatureComputationResponse(
            request_id=request.request_id,
            symbol=request.symbol,
            processing_context=request.processing_context,
            success=True,
            stats=stats,
            feature_store_metadata=feature_store_metadata,
            started_at=processing_start,
            completed_at=processing_end,
        )

    def _create_no_computation_response(
        self,
        request: FeatureComputationRequest,
        processing_start: datetime,
        existing_features_info: Optional[Dict[Timeframe, FeatureExistenceCheckResult]],
    ) -> FeatureComputationResponse:
        """Create a response when no computation was needed."""

        processing_end = datetime.now()

        features_skipped = 0
        if existing_features_info:
            for existence_result in existing_features_info.values():
                features_skipped += len(existence_result.existing_features)

        stats = FeatureProcessingStats(
            features_requested=len(request.get_enabled_features()) * len(request.target_timeframes),
            features_computed=0,
            features_skipped=features_skipped,
            features_failed=0,
            processing_duration_ms=int((processing_end - processing_start).total_seconds() * 1000),
            data_points_processed=0,
            computation_rate_per_second=0.0,
        )

        return FeatureComputationResponse(
            request_id=request.request_id,
            symbol=request.symbol,
            processing_context=request.processing_context,
            success=True,
            stats=stats,
            feature_store_metadata=FeatureStoreMetadata(),
            started_at=processing_start,
            completed_at=processing_end,
        )

    def _create_error_response(
        self, request: FeatureComputationRequest, processing_start: datetime, error_message: str
    ) -> FeatureComputationResponse:
        """Create an error response."""

        processing_end = datetime.now()

        stats = FeatureProcessingStats(
            features_requested=len(request.get_enabled_features()) * len(request.target_timeframes),
            features_computed=0,
            features_skipped=0,
            features_failed=len(request.get_enabled_features()) * len(request.target_timeframes),
            processing_duration_ms=int((processing_end - processing_start).total_seconds() * 1000),
            data_points_processed=0,
            computation_rate_per_second=0.0,
        )

        return FeatureComputationResponse(
            request_id=request.request_id,
            symbol=request.symbol,
            processing_context=request.processing_context,
            success=False,
            error_message=error_message,
            stats=stats,
            feature_store_metadata=FeatureStoreMetadata(),
            started_at=processing_start,
            completed_at=processing_end,
        )

    def _update_performance_metrics(self, response: FeatureComputationResponse) -> None:
        """Update internal performance tracking metrics."""
        self._total_features_computed += response.stats.features_computed
        self._total_processing_time_ms += response.stats.processing_duration_ms

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
#         final_result = self.context_feature_service.merge_context_features(
#             merged_result, context_features
#         )

#         logger.info("6. Preprocessing finished.")
#         return PreprocessingResult(
#             symbol_container=symbol_container,
#             computed_dataset_containers=computed_dataset_containers,
#             base_computed_container=base_computed_container,
#             other_computed_containers=other_computed_containers,
#             merged_result=merged_result,
#             context_features=context_features,
#             final_result=final_result,
#         )

#     def _merge_all_timeframes_features_together(
#         self,
#         base_computed_container: ComputedDataSetContainer,
#         other_computed_containers: list[ComputedDataSetContainer],
#     ) -> DataFrame:
#         """
#         Merges all computed features across different timeframes into a single DataFrame.

#         Returns:
#             DataFrame: The merged DataFrame with all features.
#         """
#         base_frame: DataFrame = base_computed_container.computed_dataframe
#         # Ensure base_frame has a DatetimeIndex
#         base_frame = ensure_datetime_index(base_frame, "base frame for merging")

#         delayed_tasks = []

#         for _i, container in enumerate(other_computed_containers):
#             # Ensure higher timeframe dataframe has a DatetimeIndex
#             higher_df = ensure_datetime_index(
#                 container.computed_dataframe,
#                 f"higher timeframe for {container.source_dataset.timeframe}",
#             )
#             task = delayed(self.merge_service.merge_timeframes)(
#                 base_frame.copy(), higher_df.copy()
#             )
#             delayed_tasks.append(task)

#         all_timeframes_computed_features: List[DataFrame] = dask.compute(*delayed_tasks)
#         len_bf = len(base_frame)
#         # Validate if all dataframes have same length as base_frame
#         any_length_mismatch = False
#         for i, df in enumerate(all_timeframes_computed_features):
#             len_df = len(df)
#             if len_df != len_bf:
#                 logger.error(
#                     f"DataFrame {i} has a different length ({len_df}) than the base frame ({len_bf}). Skipping merge."
#                 )
#                 any_length_mismatch = True

#         if any_length_mismatch:
#             raise ValueError(
#                 "One or more DataFrames have a different length than the base frame. Merging aborted."
#             )

#         # Merge all timeframes into the base frame using pd.concat
#         try:
#             # Ensure all DataFrames have DatetimeIndex before concatenation
#             all_dfs = [base_frame] + [
#                 ensure_datetime_index(df, f"higher timeframe result {i}")
#                 for i, df in enumerate(all_timeframes_computed_features)
#             ]
#             # Use pd.concat to merge all dataframes at once along the column axis (axis=1)
#             merged_result = pd.concat(all_dfs, axis=1)

#             # Ensure we don't have duplicate columns after concat
#             if len(merged_result.columns) != sum(len(df.columns) for df in all_dfs):
#                 logger.warning(
#                     "Detected duplicate column names during concatenation. Some data may be overwritten."
#                 )
#         except Exception as e:
#             logger.error(f"Error merging timeframes with pd.concat: {e}")
#             raise

#         return merged_result

#     def _compute_features_for_all_timeframes(
#         self,
#         datasets: List[AssetPriceDataSet],
#         symbol: str,
#     ) -> List[ComputedDataSetContainer]:
#         """
#         Computes features for all datasets in parallel.

#         Args:
#             datasets: List of AssetPriceDataSet to compute features for
#             symbol: The symbol name

#         Returns:
#             List of ComputedDataSetContainer with computed features
#         """

#         # Process each dataset (timeframe) in parallel
#         delayed_timeframe_computation = []
#         for dataset in datasets:
#             task = delayed(self._compute_features_for_dataset)(dataset, symbol)
#             delayed_timeframe_computation.append(task)

#         # Execute all processing tasks
#         processed_timeframe_containers: list[Optional[ComputedDataSetContainer]] = (
#             dask.compute(*delayed_timeframe_computation)
#         )

#         # Filter out None results
#         return [
#             container
#             for container in processed_timeframe_containers
#             if container is not None
#         ]
