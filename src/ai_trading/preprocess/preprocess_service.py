import logging
from typing import List, Optional

import dask
import pandas as pd
from pandas import DataFrame

from ai_trading.config.feature_config import FeaturesConfig
from ai_trading.data_set_utils.merge_service import MergeService
from ai_trading.data_set_utils.util import (
    ensure_datetime_time_column,
    separate_computed_datasets,
)
from ai_trading.model.computed_dataset_container import ComputedDataSetContainer
from ai_trading.model.symbol_import_container import SymbolImportContainer
from ai_trading.preprocess.feast.feast_service import FeastService
from ai_trading.preprocess.feature.feature_aggregator import FeatureAggregator
from ai_trading.preprocess.feature.feature_class_registry import FeatureClassRegistry

logger = logging.getLogger(__name__)


class PreprocessService:
    def __init__(
        self,
        features_config: FeaturesConfig,
        feature_class_registry: FeatureClassRegistry,
        feast_service: FeastService,
    ) -> None:
        """
        Initializes the PreprocessService with configuration and stateless dependencies.

        Args:
            features_config: Configuration for feature computation
            feature_class_registry: Registry of available feature classes
            feast_service: Service for feature store interaction
        """
        self.features_config = features_config
        self.feature_class_registry = feature_class_registry
        self.feast_service = feast_service

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

    def preprocess_data(self, symbol_container: SymbolImportContainer) -> DataFrame:
        """
        Preprocesses data by computing features in parallel and merging timeframes.

        Steps:
        1. Generate delayed feature computation tasks for all datasets.
        2. Execute all tasks in parallel using dask.compute.
        3. Aggregate computed features for each original dataset.
        4. Merge aggregated datasets across timeframes.

        Args:
            symbol_container: Container holding symbol datasets to process

        Returns:
            DataFrame: The final merged DataFrame with all features.
        """
        datasets = list(symbol_container.datasets)
        symbol = symbol_container.symbol

        logger.info(
            f"Starting preprocessing for symbol {symbol} with {len(datasets)} datasets."
        )
        # 1. Create FeatureAggregator instances and get delayed tasks for each dataset
        all_delayed_tasks = []
        dataset_task_indices = {}  # Map dataset index to its range of tasks
        original_datasets_map = {}  # Map dataset index to original AssetPriceDataSet
        current_task_index = 0

        for i, dataset in enumerate(datasets):
            logger.debug(
                f"Generating feature tasks for dataset {i}: Symbol={symbol}, Timeframe={dataset.timeframe}"
            )
            # Pass FeastService to FeatureAggregator
            feature_aggregator = FeatureAggregator(
                config=self.features_config,
                class_registry=self.feature_class_registry,
                feast_service=self.feast_service,
            )
            # compute now takes asset_data and symbol parameters
            dataset_delayed_tasks = feature_aggregator.compute(
                asset_data=dataset, symbol=symbol
            )

            start_index = current_task_index
            all_delayed_tasks.extend(dataset_delayed_tasks)
            end_index = len(all_delayed_tasks)
            dataset_task_indices[i] = (start_index, end_index)
            original_datasets_map[i] = dataset  # Store original dataset
            current_task_index = end_index
            logger.debug(f"Added {len(dataset_delayed_tasks)} tasks for dataset {i}.")

        computed_dataset_containers = (
            []
        )  # Initialize the list to avoid usage before definition
        if not all_delayed_tasks:
            logger.warning(
                "No feature computation tasks were generated. Proceeding without features."
            )
            computed_dataset_containers = [
                ComputedDataSetContainer(ds, ds.asset_price_dataset.copy())
                for ds in datasets
            ]
        else:
            # 2. Compute all delayed feature tasks in parallel
            logger.info(
                f"Executing {len(all_delayed_tasks)} feature tasks in parallel..."
            )
            computed_feature_dfs: List[Optional[DataFrame]] = list(
                dask.compute(*all_delayed_tasks)[0]
            )  # dask.compute returns a tuple
            logger.info("Finished parallel feature computation.")

            # 3. Aggregate results per original dataset
            computed_dataset_containers.clear()
            for i, original_dataset in original_datasets_map.items():
                dataset_info = (
                    f"Symbol={symbol}, Timeframe={original_dataset.timeframe}"
                )
                logger.debug(f"Aggregating features for dataset {i} ({dataset_info}).")
                start_idx, end_idx = dataset_task_indices[i]
                dataset_feature_results = [
                    df
                    for df in computed_feature_dfs[start_idx:end_idx]
                    if df is not None and not df.empty
                ]
                logger.debug(
                    f"Found {len(dataset_feature_results)} valid feature DataFrames for dataset {i}."
                )

                final_df_for_dataset_indexed = self._prepare_dataframe_for_join(
                    original_dataset.asset_price_dataset,
                    f"original data ({dataset_info})",
                )

                if final_df_for_dataset_indexed is None:
                    logger.error(
                        f"Could not prepare original data for dataset {i} ({dataset_info}) for joining. Skipping dataset."
                    )
                    continue

                if dataset_feature_results:
                    dfs_to_join: List[DataFrame] = []
                    for idx, feat_df in enumerate(dataset_feature_results):
                        prepared_feat_df = self._prepare_dataframe_for_join(
                            feat_df, f"feature set {idx} ({dataset_info})"
                        )
                        if prepared_feat_df is not None:
                            cols_to_add = prepared_feat_df.columns
                            existing_cols = final_df_for_dataset_indexed.columns.union(
                                pd.Index(
                                    [col for df in dfs_to_join for col in df.columns]
                                )
                            )
                            duplicates = cols_to_add.intersection(existing_cols)
                            if not duplicates.empty:
                                logger.warning(
                                    f"Duplicate columns found for {dataset_info} in feature set {idx}: {duplicates.tolist()}. Dropping duplicates from feature set."
                                )
                                prepared_feat_df = prepared_feat_df.drop(
                                    columns=duplicates
                                )

                            if not prepared_feat_df.empty:
                                dfs_to_join.append(prepared_feat_df)
                        else:
                            logger.warning(
                                f"Could not prepare feature set {idx} for dataset {i} ({dataset_info}) for joining."
                            )

                    if dfs_to_join:
                        logger.debug(
                            f"Joining {len(dfs_to_join)} feature sets to dataset {i}."
                        )
                        try:
                            final_df_for_dataset_indexed = (
                                final_df_for_dataset_indexed.join(
                                    pd.concat(dfs_to_join, axis=1), how="left"
                                )
                            )
                        except Exception as e:
                            logger.error(
                                f"Error joining features for dataset {i} ({dataset_info}): {e}",
                                exc_info=True,
                            )
                            final_df_for_dataset_indexed = (
                                self._prepare_dataframe_for_join(
                                    original_dataset.asset_price_dataset,
                                    f"original data ({dataset_info}) - recovery",
                                )
                            )
                            if final_df_for_dataset_indexed is None:
                                continue

                final_df_for_dataset = final_df_for_dataset_indexed.reset_index()

                computed_dataset_containers.append(
                    ComputedDataSetContainer(original_dataset, final_df_for_dataset)
                )
                logger.debug(
                    f"Finished aggregation for dataset {i}. Final shape: {final_df_for_dataset.shape}"
                )

        if not computed_dataset_containers:
            logger.error("No datasets could be processed successfully.")
            return pd.DataFrame()

        # 4. Separate base and other datasets
        logger.info("Separating base and other datasets.")
        base_computed_container, other_computed_containers = separate_computed_datasets(
            computed_dataset_containers
        )

        if base_computed_container is None:
            logger.error(
                "Base dataset is missing after feature computation/aggregation."
            )
            return pd.DataFrame()

        # 5. Merge timeframes
        logger.info("Starting timeframe merging.")
        merged_result: DataFrame = base_computed_container.computed_dataframe

        for i, container in enumerate(other_computed_containers):
            logger.debug(
                f"Merging other dataset {i} (Timeframe: {container.source_dataset.timeframe}) into base."
            )
            merger = MergeService(merged_result, container.computed_dataframe)
            merged_result = merger.merge_timeframes()
            logger.debug(f"Shape after merging dataset {i}: {merged_result.shape}")

        logger.info("Preprocessing finished.")
        return merged_result
