import logging
from typing import List, Optional

import dask
from dask import delayed
from pandas import DataFrame

from drl_trading_framework.common.config.feature_config import FeaturesConfig
from drl_trading_framework.common.model.asset_price_dataset import AssetPriceDataSet
from drl_trading_framework.common.model.computed_dataset_container import (
    ComputedDataSetContainer,
)
from drl_trading_framework.common.model.symbol_import_container import (
    SymbolImportContainer,
)
from drl_trading_framework.preprocess.data_set_utils.context_feature_service import (
    ContextFeatureService,
)
from drl_trading_framework.preprocess.data_set_utils.merge_service import (
    MergeServiceInterface,
)
from drl_trading_framework.preprocess.data_set_utils.util import (
    ensure_datetime_index,
    separate_asset_price_datasets,
    separate_computed_datasets,
)
from drl_trading_framework.preprocess.feature.feature_aggregator import (
    FeatureAggregatorInterface,
)
from drl_trading_framework.preprocess.feature.feature_class_registry import (
    FeatureClassRegistry,
)

logger = logging.getLogger(__name__)


class PreprocessService:
    def __init__(
        self,
        features_config: FeaturesConfig,
        feature_class_registry: FeatureClassRegistry,
        feature_aggregator: FeatureAggregatorInterface,
        merge_service: MergeServiceInterface,
        context_feature_service: ContextFeatureService,
    ) -> None:
        """
        Initializes the PreprocessService with configuration and stateless dependencies.

        Args:
            features_config: Configuration for feature computation
            feature_class_registry: Registry of available feature classes
            feature_aggregator: Service for feature aggregation and computation
            merge_service: Service for merging timeframes
            context_feature_service: Service for handling context-related features
        """
        self.features_config = features_config
        self.feature_class_registry = feature_class_registry
        self.feature_aggregator = feature_aggregator
        self.merge_service = merge_service
        self.context_feature_service = context_feature_service

    def _prepare_dataframe_for_join(
        self, df: DataFrame, dataset_info: str
    ) -> Optional[DataFrame]:
        """Ensures DataFrame has a DatetimeIndex for efficient joining operations."""
        if df is None or df.empty:
            return None
        try:
            # Use the ensure_datetime_index utility function
            return ensure_datetime_index(df, dataset_info)
        except ValueError as ve:
            logger.error(f"Failed to ensure DatetimeIndex for {dataset_info}: {ve}")
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error creating DatetimeIndex for {dataset_info}: {e}"
            )
            return None

    def _compute_features_for_dataset(
        self, dataset: AssetPriceDataSet, symbol: str
    ) -> Optional[ComputedDataSetContainer]:
        """
        Computes features for a single dataset using the feature aggregator.

        Args:
            dataset: The dataset to compute features for
            symbol: The symbol name

        Returns:
            ComputedDataSetContainer containing the source dataset and computed features,
            or None if feature computation failed
        """
        logger.info(f"Computing features for {symbol} {dataset.timeframe}")

        # Get delayed computation tasks from feature aggregator
        delayed_tasks = self.feature_aggregator.compute(
            asset_data=dataset, symbol=symbol
        )

        # Skip if no features to compute
        if not delayed_tasks:
            logger.warning(
                f"No feature computation tasks for {symbol} {dataset.timeframe}"
            )
            return None

        # Execute all tasks in parallel
        computed_feature_dfs = dask.compute(*delayed_tasks)

        # Filter out None results
        valid_feature_dfs = [df for df in computed_feature_dfs if df is not None]

        # Skip if no valid results
        if not valid_feature_dfs:
            logger.warning(
                f"No valid features computed for {symbol} {dataset.timeframe}"
            )
            return None

        # Ensure all DataFrames have a DatetimeIndex
        valid_feature_dfs = [
            ensure_datetime_index(df, f"feature dataset {i} for {symbol}")
            for i, df in enumerate(valid_feature_dfs)
        ]

        # Start with first dataframe
        merged_features = valid_feature_dfs[0]

        # Merge remaining dataframes - use index-based merge for better performance
        for i, feature_df in enumerate(valid_feature_dfs[1:], 1):
            try:
                merged_features = merged_features.join(feature_df, how="outer")
            except Exception as e:
                logger.error(
                    f"Error merging feature dataframe {i} for {symbol} {dataset.timeframe}: {e}"
                )
                # Continue with what we have

        # Return container with source dataset and computed features
        return ComputedDataSetContainer(
            source_dataset=dataset, computed_dataframe=merged_features
        )

    def preprocess_data(self, symbol_container: SymbolImportContainer) -> DataFrame:
        """
        Preprocesses data by computing features in parallel and merging timeframes.

        Steps:
        1. Generate delayed feature computation tasks for all datasets.
        2. Execute all tasks in parallel using dask.compute.
        3. Aggregate computed features for each original dataset.
        4. Merge aggregated datasets across timeframes.
        5. Add context-related features required by the trading environment.

        Args:
            symbol_container: Container holding symbol datasets to process

        Returns:
            DataFrame: The final merged DataFrame with all features and a DatetimeIndex.
        """
        datasets: List[AssetPriceDataSet] = symbol_container.datasets
        base_dataset, _ = separate_asset_price_datasets(datasets)
        symbol = symbol_container.symbol

        logger.info(
            f"Starting preprocessing for symbol {symbol} with {len(datasets)} datasets."
        )

        # 3. Aggregate results
        computed_dataset_containers: List[ComputedDataSetContainer] = []

        # Process each dataset (timeframe) in parallel
        delayed_processing_tasks = []
        for dataset in datasets:
            task = delayed(self._compute_features_for_dataset)(dataset, symbol)
            delayed_processing_tasks.append(task)

        # Execute all processing tasks
        processed_containers = dask.compute(*delayed_processing_tasks)

        # Filter out None results
        computed_dataset_containers = [
            container for container in processed_containers if container is not None
        ]

        # Check if we have at least one valid container
        if not computed_dataset_containers:
            raise ValueError("No valid computed datasets were produced")

        # 4. Separate base and other datasets
        logger.info("Separating base and other datasets.")
        base_computed_container, other_computed_containers = separate_computed_datasets(
            computed_dataset_containers
        )

        # 5. Merge timeframes
        logger.info("Starting timeframe merging.")
        base_frame: DataFrame = base_computed_container.computed_dataframe
        # Ensure base_frame has a DatetimeIndex
        base_frame = ensure_datetime_index(base_frame, "base frame for merging")

        delayed_tasks = []

        for _i, container in enumerate(other_computed_containers):
            # Ensure higher timeframe dataframe has a DatetimeIndex
            higher_df = ensure_datetime_index(
                container.computed_dataframe,
                f"higher timeframe for {container.source_dataset.timeframe}",
            )
            task = delayed(self.merge_service.merge_timeframes)(
                base_frame.copy(), higher_df.copy()
            )
            delayed_tasks.append(task)

        all_timeframes_computed_features: List[DataFrame] = dask.compute(*delayed_tasks)

        len_bf = len(base_frame)
        # Validate if all dataframes have same length as base_frame
        any_length_mismatch = False
        for i, df in enumerate(all_timeframes_computed_features):
            len_df = len(df)
            if len_df != len_bf:
                logger.error(
                    f"DataFrame {i} has a different length ({len_df}) than the base frame ({len_bf}). Skipping merge."
                )
                any_length_mismatch = True

        if any_length_mismatch:
            raise ValueError(
                "One or more DataFrames have a different length than the base frame. Merging aborted."
            )

        # Merge all timeframes into the base frame using index-based operations
        merged_result: DataFrame = base_frame
        for i, df in enumerate(all_timeframes_computed_features):
            try:
                # Ensure the higher timeframe result has a DatetimeIndex
                df = ensure_datetime_index(df, f"higher timeframe result {i}")
                # Join on index rather than using a column
                merged_result = merged_result.join(df, how="left")
            except Exception as e:
                logger.error(f"Error merging timeframe {i}: {e}")
                raise

        # 6. Merge context-related features using the dedicated service
        logger.info("Preparing context-related features.")
        context_features = self.context_feature_service.prepare_context_features(
            base_dataset
        )

        logger.info("Merging context-related features with computed features.")
        final_result = self.context_feature_service.merge_context_features(
            merged_result, context_features
        )

        logger.info("Preprocessing finished.")
        return final_result
