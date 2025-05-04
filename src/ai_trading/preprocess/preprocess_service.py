import logging
from typing import List, Optional

import dask
from dask import delayed
from pandas import DataFrame

from ai_trading.config.feature_config import FeaturesConfig
from ai_trading.data_set_utils.context_feature_service import ContextFeatureService
from ai_trading.data_set_utils.merge_service import MergeServiceInterface
from ai_trading.data_set_utils.util import (
    ensure_datetime_time_column,
    separate_asset_price_datasets,
    separate_computed_datasets,
)
from ai_trading.model.asset_price_dataset import AssetPriceDataSet
from ai_trading.model.computed_dataset_container import ComputedDataSetContainer
from ai_trading.model.symbol_import_container import SymbolImportContainer
from ai_trading.preprocess.feature.feature_aggregator import FeatureAggregatorInterface
from ai_trading.preprocess.feature.feature_class_registry import FeatureClassRegistry

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
        """Ensures DataFrame has a valid 'Time' column and sets it as index."""
        if df is None or df.empty:
            return None
        try:
            # First, ensure the 'Time' column is valid using the utility function
            df_with_time = ensure_datetime_time_column(df, dataset_info)
            # Now, set the validated 'Time' column as index
            return df_with_time.set_index("Time")
        except ValueError as ve:  # Catch errors from ensure_datetime_time_column
            logger.error(
                f"Failed to ensure valid 'Time' column for {dataset_info}: {ve}"
            )
            return None
        except Exception as e:  # Catch errors from set_index
            logger.error(f"Failed to set 'Time' index for {dataset_info}: {e}")
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

        # Start with first dataframe
        merged_features = valid_feature_dfs[0]

        # Merge remaining dataframes
        for i, feature_df in enumerate(valid_feature_dfs[1:], 1):
            try:
                merged_features = merged_features.merge(
                    feature_df, on="Time", how="outer"
                )
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
            DataFrame: The final merged DataFrame with all features.
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

        delayed_tasks = []

        for _i, container in enumerate(other_computed_containers):
            task = delayed(self.merge_service.merge_timeframes)(
                base_frame.copy(), container.computed_dataframe.copy()
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

        # Merge all timeframes into the base frame
        merged_result: DataFrame = base_frame
        for _i, df in enumerate(all_timeframes_computed_features):
            merged_result = merged_result.merge(df, on="Time", how="left")

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
