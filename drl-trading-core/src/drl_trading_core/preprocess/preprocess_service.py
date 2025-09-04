import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import dask
import pandas as pd
from dask import delayed
from drl_trading_common.config.feature_config import FeaturesConfig
from drl_trading_common.utils import ensure_datetime_index
from drl_trading_common.interface.feature.context_feature_service_interface import (
    ContextFeatureServiceInterface,
)
from drl_trading_common.interface.feature.feature_factory_interface import (
    IFeatureFactory,
)
from injector import inject
from pandas import DataFrame

from drl_trading_core.common.model.asset_price_dataset import AssetPriceDataSet
from drl_trading_core.common.model.computed_dataset_container import (
    ComputedDataSetContainer,
)
from drl_trading_core.common.model.preprocessing_result import PreprocessingResult
from drl_trading_core.common.model.symbol_import_container import (
    SymbolImportContainer,
)
from drl_trading_preprocess.core.service.computing_service import IFeatureComputer
from drl_trading_core.preprocess.data_set_utils.merge_service import (
    MergeServiceInterface,
)
from drl_trading_core.preprocess.data_set_utils.util import (
    separate_asset_price_datasets,
    separate_computed_datasets,
)

logger = logging.getLogger(__name__)


# Define interface for preprocessing service
class PreprocessServiceInterface(ABC):
    """
    Interface defining the contract for preprocessing services.
    """

    @abstractmethod
    def preprocess_data(
        self, symbol_container: SymbolImportContainer
    ) -> PreprocessingResult:
        """
        Preprocesses the data for a given symbol by computing features and merging datasets.

        Args:
            symbol_container (SymbolImportContainer): Container with symbol and datasets

        Returns:
            PreprocessingResult: Final preprocessing results with computed features.
        """
        ...


@inject
class PreprocessService(PreprocessServiceInterface):

    def __init__(
        self,
        features_config: FeaturesConfig,
        feature_factory: IFeatureFactory,
        merge_service: MergeServiceInterface,
        context_feature_service: ContextFeatureServiceInterface,
        computer: IFeatureComputer,
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
        self.feature_class_registry = feature_factory
        self.merge_service = merge_service
        self.context_feature_service = context_feature_service
        self.feature_computer = computer

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
        delayed_tasks = self.feature_computer.compute_batch(dataset.asset_price_dataset)

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

    def preprocess_data(
        self, symbol_container: SymbolImportContainer
    ) -> PreprocessingResult:
        """
        Preprocesses the data for a given symbol by computing features and merging datasets.

        Args:
            symbol_container (SymbolImportContainer): Container with symbol and datasets

        Raises:
            ValueError: If no valid computed datasets were produced

        Returns:
            PreprocessingResult: Object containing all intermediate and final preprocessing results
        """
        datasets: List[AssetPriceDataSet] = symbol_container.datasets
        base_dataset, _ = separate_asset_price_datasets(datasets)
        symbol = symbol_container.symbol

        logger.info(
            f"Starting preprocessing for symbol {symbol} with {len(datasets)} datasets."
        )

        # 1. Aggregate results
        logger.info(f"Computing features for all timeframes for {symbol}.")
        computed_dataset_containers: List[ComputedDataSetContainer] = (
            self._compute_features_for_all_timeframes(datasets, symbol)
        )

        # Check if we have at least one valid container
        if not computed_dataset_containers:
            raise ValueError("No valid computed datasets were produced")
        # 4. Separate base and other datasets
        logger.info("2. Separating base and other datasets.")
        base_computed_container, other_computed_containers = separate_computed_datasets(
            computed_dataset_containers
        )

        # Check if base_computed_container was found
        if base_computed_container is None:
            raise ValueError("No base dataset found after feature computation")

        # 5. Merge timeframes
        logger.info("3. Starting feature merging across all timeframes.")
        merged_result = self._merge_all_timeframes_features_together(
            base_computed_container, other_computed_containers
        )

        # 6. Merge context-related features using the dedicated service
        logger.info("4. Preparing context-related features.")
        context_features = self.context_feature_service.prepare_context_features(
            base_dataset
        )

        logger.info("5. Merging context-related features with computed features.")
        final_result = self.context_feature_service.merge_context_features(
            merged_result, context_features
        )

        logger.info("6. Preprocessing finished.")
        return PreprocessingResult(
            symbol_container=symbol_container,
            computed_dataset_containers=computed_dataset_containers,
            base_computed_container=base_computed_container,
            other_computed_containers=other_computed_containers,
            merged_result=merged_result,
            context_features=context_features,
            final_result=final_result,
        )

    def _merge_all_timeframes_features_together(
        self,
        base_computed_container: ComputedDataSetContainer,
        other_computed_containers: list[ComputedDataSetContainer],
    ) -> DataFrame:
        """
        Merges all computed features across different timeframes into a single DataFrame.

        Returns:
            DataFrame: The merged DataFrame with all features.
        """
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

        # Merge all timeframes into the base frame using pd.concat
        try:
            # Ensure all DataFrames have DatetimeIndex before concatenation
            all_dfs = [base_frame] + [
                ensure_datetime_index(df, f"higher timeframe result {i}")
                for i, df in enumerate(all_timeframes_computed_features)
            ]
            # Use pd.concat to merge all dataframes at once along the column axis (axis=1)
            merged_result = pd.concat(all_dfs, axis=1)

            # Ensure we don't have duplicate columns after concat
            if len(merged_result.columns) != sum(len(df.columns) for df in all_dfs):
                logger.warning(
                    "Detected duplicate column names during concatenation. Some data may be overwritten."
                )
        except Exception as e:
            logger.error(f"Error merging timeframes with pd.concat: {e}")
            raise

        return merged_result

    def _compute_features_for_all_timeframes(
        self,
        datasets: List[AssetPriceDataSet],
        symbol: str,
    ) -> List[ComputedDataSetContainer]:
        """
        Computes features for all datasets in parallel.

        Args:
            datasets: List of AssetPriceDataSet to compute features for
            symbol: The symbol name

        Returns:
            List of ComputedDataSetContainer with computed features
        """

        # Process each dataset (timeframe) in parallel
        delayed_timeframe_computation = []
        for dataset in datasets:
            task = delayed(self._compute_features_for_dataset)(dataset, symbol)
            delayed_timeframe_computation.append(task)

        # Execute all processing tasks
        processed_timeframe_containers: list[Optional[ComputedDataSetContainer]] = (
            dask.compute(*delayed_timeframe_computation)
        )

        # Filter out None results
        return [
            container
            for container in processed_timeframe_containers
            if container is not None
        ]
