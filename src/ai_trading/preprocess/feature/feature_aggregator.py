import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd
from dask import delayed
from dask.delayed import Delayed
from pandas import DataFrame

from ai_trading.config.feature_config import (
    BaseParameterSetConfig,
    FeatureDefinition,
    FeaturesConfig,
)
from ai_trading.data_set_utils.util import ensure_datetime_time_column
from ai_trading.model.asset_price_dataset import AssetPriceDataSet
from ai_trading.preprocess.feast.feast_service import FeastServiceInterface
from ai_trading.preprocess.feature.feature_class_registry import FeatureClassRegistry

logger = logging.getLogger(__name__)


class FeatureAggregatorInterface(ABC):
    """
    Interface defining the contract for feature aggregation operations.

    Implementations of this interface are responsible for:
    1. Defining computation tasks for features based on configuration.
    2. Using the feature store to retrieve previously computed features within tasks.
    3. Storing newly computed features in the feature store within tasks.
    4. Renaming feature columns according to a consistent convention.
    """

    @abstractmethod
    def compute(self, asset_data: AssetPriceDataSet, symbol: str) -> List[Delayed]:
        """
        Generates a list of delayed tasks for computing or retrieving features.

        Each task corresponds to one feature definition and one parameter set.
        The task, when executed, will return a DataFrame containing the 'Time'
        column and the renamed feature columns, or None if skipped/failed.

        Args:
            asset_data: Dataset containing asset price information.
            symbol: The trading symbol being processed.

        Returns:
            List[dask.delayed]: A list of delayed objects ready for computation.
        """
        pass


class FeatureAggregator(FeatureAggregatorInterface):
    """
    Aggregates and computes features for asset price datasets using delayed execution.

    This class is responsible for:
    1. Defining computation tasks for features based on configuration.
    2. Using the feature store to retrieve previously computed features within tasks.
    3. Storing newly computed features in the feature store within tasks.
    4. Renaming feature columns according to the convention: featureName_paramHash_subFeatureName.
    """

    def __init__(
        self,
        config: FeaturesConfig,
        class_registry: FeatureClassRegistry,
        feast_service: FeastServiceInterface,
    ) -> None:
        """
        Initialize the FeatureAggregator with configuration and services.

        Args:
            config: Configuration for feature definitions.
            class_registry: Registry of feature classes.
            feast_service: Service for interacting with the Feast feature store.
        """
        self.config = config
        self.class_registry = class_registry
        self.feast_service = feast_service

    def _compute_or_get_single_feature(
        self,
        feature_def: FeatureDefinition,
        param_set: BaseParameterSetConfig,
        original_df: DataFrame,
        symbol: str,
        asset_data: AssetPriceDataSet,
    ) -> Optional[DataFrame]:
        """
        Computes or retrieves a single feature for a given parameter set.
        Handles caching and column renaming. This method is intended to be
        wrapped by dask.delayed.

        Args:
            feature_def: The definition of the feature.
            param_set: The specific parameter set to use.
            original_df: The dataset to compute features on.
            symbol: The trading symbol being processed.
            asset_data: The asset price dataset containing metadata.

        Returns:
            DataFrame with the 'Time' column and the computed/retrieved feature
            columns, renamed according to convention (featureName_paramHash_subFeatureName),
            or None if the feature/param_set is disabled or computation fails.
        """
        if not feature_def.enabled or not param_set.enabled:
            return None

        # Create feature instance
        feature_class = self.class_registry.feature_class_map[feature_def.name]
        # Pass the prepared original DataFrame copy to the feature instance
        feature_instance = feature_class(source=original_df, config=param_set)

        feature_name = feature_instance.get_feature_name()
        param_hash: str = param_set.hash_id
        sub_feature_names = feature_instance.get_sub_features_names()

        # Try to get features from store first
        historical_features: Optional[DataFrame] = None
        if self.feast_service.is_enabled():
            historical_features = self.feast_service.get_historical_features(
                feature_name=feature_name,
                param_hash=param_hash,
                sub_feature_names=sub_feature_names,
                asset_data=asset_data,
                symbol=symbol,
            )

        feature_df: Optional[DataFrame] = None
        if historical_features is not None and not historical_features.empty:
            logger.debug(
                f"Using cached features for {feature_name} with params hash {param_hash}"
            )
            # Ensure 'Time' column is present in cached data
            if "Time" not in historical_features.columns:
                if isinstance(historical_features.index, pd.DatetimeIndex):
                    historical_features = historical_features.reset_index()
                else:
                    logger.warning(
                        f"Cached features for {feature_name} (hash: {param_hash}) missing 'Time'. Skipping."
                    )
                    return None  # Cannot use without Time for joining
            feature_df = historical_features
        else:
            # Compute features if not found in store or store disabled
            logger.info(
                f"Computing features for {feature_name} with params hash {param_hash}"
            )
            try:
                computed_df = feature_instance.compute()
            except Exception as e:
                logger.error(
                    f"Error computing {feature_name} with hash {param_hash}: {e}",
                    exc_info=True,
                )
                return None  # Skip this feature/param set on error

            # Ensure 'Time' column exists after computation
            if "Time" not in computed_df.columns:
                if isinstance(computed_df.index, pd.DatetimeIndex):
                    computed_df = computed_df.reset_index()
                else:
                    logger.error(
                        f"Computed DataFrame for {feature_name} (hash: {param_hash}) lacks 'Time' column or DatetimeIndex. Skipping."
                    )
                    return None  # Cannot proceed without Time

            # Store computed features if feature store is enabled
            if self.feast_service.is_enabled():
                try:
                    self.feast_service.store_computed_features(
                        feature_df=computed_df.copy(),  # Pass a copy
                        feature_name=feature_name,
                        param_hash=param_hash,
                        sub_feature_names=sub_feature_names,
                        asset_data=asset_data,
                        symbol=symbol,
                    )
                except Exception as e:
                    logger.error(
                        f"Error storing features for {feature_name} (hash: {param_hash}): {e}",
                        exc_info=True,
                    )
                    # Continue with the computed data even if storing failed

            feature_df = computed_df

        if feature_df is None or feature_df.empty:
            logger.warning(
                f"No features generated or retrieved for {feature_name} with params hash {param_hash}"
            )
            return None

        # --- Column Renaming and Selection ---
        # Ensure 'Time' column is present before renaming (should be guaranteed by ensure_datetime_time_column)
        if "Time" not in feature_df.columns:
            # This check might still be useful if compute() or get_historical_features() returns malformed data
            logger.error(
                f"Feature DataFrame for {feature_name} (hash: {param_hash}) missing 'Time' column before renaming. Skipping."
            )
            return None

        rename_map = {}
        columns_to_keep = ["Time"]  # Always keep 'Time'

        for sub_feature in sub_feature_names:
            if sub_feature in feature_df.columns:
                new_name = f"{feature_name}_{param_hash}_{sub_feature}"
                rename_map[sub_feature] = new_name
                columns_to_keep.append(sub_feature)  # Keep original name for selection
            else:
                logger.warning(
                    f"Sub-feature '{sub_feature}' not found in computed/cached df for {feature_name} (hash: {param_hash})."
                )

        # Select only the 'Time' column and the sub-features to be renamed
        feature_df_selected = feature_df[columns_to_keep]

        # Rename the selected columns
        feature_df_renamed = feature_df_selected.rename(columns=rename_map)

        # Final check for 'Time' column after operations
        if "Time" not in feature_df_renamed.columns:
            logger.critical(
                f"Lost 'Time' column during processing for {feature_name} (hash: {param_hash}). This should not happen."
            )
            return None

        return feature_df_renamed

    def compute(self, asset_data: AssetPriceDataSet, symbol: str) -> List[Delayed]:
        """
        Generates a list of delayed tasks for computing or retrieving features.

        Each task corresponds to one feature definition and one parameter set.
        The task, when executed, will return a DataFrame containing the 'Time'
        column and the renamed feature columns, or None if skipped/failed.

        Args:
            asset_data: Dataset containing asset price information.
            symbol: The trading symbol being processed.

        Returns:
            List[dask.delayed]: A list of delayed objects ready for computation.
        """
        delayed_tasks = []

        # Use the utility function to prepare the original DataFrame
        original_df = ensure_datetime_time_column(
            asset_data.asset_price_dataset, f"original data for symbol {symbol}"
        )

        for feature_def in self.config.feature_definitions:
            if not feature_def.enabled:
                continue

            for param_set in feature_def.parsed_parameter_sets:
                if not param_set.enabled:
                    continue

                # Create a delayed task for each feature/param combination
                # Pass copies of feature_def and param_set if they might be mutated,
                # though usually config objects are treated as immutable.
                task = delayed(self._compute_or_get_single_feature)(
                    feature_def, param_set, original_df, symbol, asset_data
                )
                delayed_tasks.append(task)

        logger.info(
            f"Generated {len(delayed_tasks)} delayed feature computation tasks for symbol {symbol}."
        )
        return delayed_tasks
