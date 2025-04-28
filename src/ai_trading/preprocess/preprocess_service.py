import logging
from typing import Optional

import dask
import pandas_ta as ta
from dask import delayed
from pandas import DataFrame

from ai_trading.config.feature_config import FeaturesConfig
from ai_trading.data_set_utils.merge_service import MergeService
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
        merge_service: MergeService,
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
        self.feature_aggregator = feature_aggregator
        self.merge_service = merge_service

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

    def _prepare_context_related_features(
        self, base_dataset: AssetPriceDataSet
    ) -> DataFrame:
        """Prepares a DataFrame with only the essential context-related features needed by the trading environment.

        This method extracts only the specific columns that are required by the gym environment
        for proper functioning, such as price data, volume, and technical indicators like ATR.

        Args:
            base_dataset (AssetPriceDataSet): The base dataset containing OHLC data and computed indicators.

        Returns:
            DataFrame: A DataFrame containing only the columns needed for the trading environment context.

        Raises:
            ValueError: If essential columns are missing from the base dataset.
        """
        # Get the DataFrame from the base dataset
        df = base_dataset.asset_price_dataset.copy()

        # Define required columns for the trading environment
        required_columns = ["Time", "Open", "High", "Low", "Close", "Volume"]

        # Define optional but important columns if they exist
        context_columns = ["Atr"]

        # Verify all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Essential columns missing from base dataset: {missing_columns}"
            )

        # Start with required columns
        selected_columns = required_columns.copy()

        # Add any optional context columns that exist in the DataFrame
        for col in context_columns:
            if col in df.columns:
                selected_columns.append(col)

        # Calculate ATR with a standard period
        df["Atr"] = ta.atr(df["High"], df["Low"], df["Close"], timeperiod=14)
        selected_columns.append("Atr")

        # Return only the selected columns
        return df[selected_columns]

    def _merge_context_related_features(
        self, computed_base_dataframe: DataFrame, context_enriched_dataframe: DataFrame
    ) -> DataFrame:
        """Merges context-related features into the computed base DataFrame.

        Args:
            computed_base_dataframe (DataFrame): The DataFrame containing computed features.
            context_enriched_dataframe (DataFrame): The DataFrame containing context-related features.

        Returns:
            DataFrame: The merged DataFrame with context-related features.
        """
        return computed_base_dataframe.merge(
            context_enriched_dataframe,
            on="Time",
            how="left",
        )

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
        datasets: list[AssetPriceDataSet] = symbol_container.datasets
        base_dataset, _ = separate_asset_price_datasets(datasets)
        symbol = symbol_container.symbol

        logger.info(
            f"Starting preprocessing for symbol {symbol} with {len(datasets)} datasets."
        )

        # 3. Aggregate results per original dataset
        computed_dataset_containers: list[ComputedDataSetContainer] = []

        # TODO: let agent to that

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

        all_timeframes_computed_features: list[DataFrame] = dask.compute(*delayed_tasks)

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

        # 6. Merge context-related features
        self._merge_context_related_features(
            merged_result, self._prepare_context_related_features(base_dataset)
        )

        logger.info("Preprocessing finished.")
        return merged_result
