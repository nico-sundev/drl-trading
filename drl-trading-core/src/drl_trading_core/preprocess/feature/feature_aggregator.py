import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd
from dask import delayed
from dask.delayed import Delayed
from drl_trading_common.config.feature_config import (
    BaseParameterSetConfig,
    FeatureDefinition,
    FeaturesConfig,
)
from drl_trading_common.interfaces.indicator.technical_indicator_facade_interface import (
    TechnicalIndicatorFactoryInterface,
)
from drl_trading_common.utils import ensure_datetime_index
from injector import inject
from pandas import DataFrame

from drl_trading_core.common.model.asset_price_dataset import AssetPriceDataSet
from drl_trading_core.preprocess.feast.feast_service import FeastServiceInterface
from drl_trading_core.preprocess.feature.feature_factory import (
    FeatureFactoryInterface,
)

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
        The task, when executed, will return a DataFrame with DatetimeIndex named 'Time'
        and the renamed feature columns, or None if skipped/failed.

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
    4. Renaming feature columns according to the convention: featureName_paramString_subFeatureName.
    """


    @inject
    def __init__(
        self,
        config: FeaturesConfig,
        feature_factory: FeatureFactoryInterface,
        feast_service: FeastServiceInterface,
        technical_indicator_factory: TechnicalIndicatorFactoryInterface,
    ) -> None:
        """
        Initialize the FeatureAggregator with configuration and services.

        Args:
            config: Configuration for feature definitions.
            feature_factory: Factory interface for creating feature instances.
            feast_service: Service for interacting with the Feast feature store.
        """
        self.config = config
        self.feature_factory = feature_factory
        self.feast_service = feast_service
        self.technical_indicator_factory = technical_indicator_factory

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
            DataFrame with DatetimeIndex named 'Time' and the computed/retrieved feature
            columns, renamed according to convention, or None if the feature/param_set
            is disabled or computation fails.
        """
        if not feature_def.enabled or not param_set.enabled:
            return None

        # Create metrics service for this asset_data timeframe
        indicator_service = self.technical_indicator_factory.create()

        # Create feature instance using the factory
        feature_instance = self.feature_factory.create_feature(
            feature_name=feature_def.name,
            source_data=original_df,
            config=param_set,
            indicators_service=indicator_service,
        )

        if feature_instance is None:
            logger.error(
                f"Failed to create feature instance for '{feature_def.name}'. Skipping."
            )
            return None

        feature_name = feature_instance.get_feature_name()
        param_hash: str = param_set.hash_id()
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
            # Ensure we have a DatetimeIndex in cached data
            if not isinstance(historical_features.index, pd.DatetimeIndex):
                if "Time" in historical_features.columns:
                    # Try to convert Time column to DatetimeIndex
                    try:
                        historical_features["Time"] = pd.to_datetime(
                            historical_features["Time"]
                        )
                        historical_features = historical_features.set_index("Time")
                    except Exception:
                        logger.warning(
                            f"Cached features for {feature_name} (hash: {param_hash}) has invalid 'Time' column. Skipping."
                        )
                        return None  # Cannot use without proper time index
                else:
                    logger.warning(
                        f"Cached features for {feature_name} (hash: {param_hash}) missing DatetimeIndex. Skipping."
                    )
                    return None  # Cannot use without time index for joining
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

            # Ensure we have a DatetimeIndex after computation
            if not isinstance(computed_df.index, pd.DatetimeIndex):
                if "Time" in computed_df.columns:
                    # Try to convert Time column to DatetimeIndex
                    try:
                        computed_df["Time"] = pd.to_datetime(computed_df["Time"])
                        computed_df = computed_df.set_index("Time")
                    except Exception as e:
                        logger.error(
                            f"Cannot convert 'Time' to DatetimeIndex for {feature_name} (hash: {param_hash}): {e}"
                        )
                        return None  # Cannot proceed without proper time index
                else:
                    logger.error(
                        f"Computed DataFrame for {feature_name} (hash: {param_hash}) lacks DatetimeIndex. Skipping."
                    )
                    return None  # Cannot proceed without time index

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

        # --- Column Renaming and Selection ---        # Ensure we have a DatetimeIndex before renaming
        if not isinstance(feature_df.index, pd.DatetimeIndex):
            logger.error(
                f"Feature DataFrame for {feature_name} (hash: {param_hash}) missing DatetimeIndex before renaming. Skipping."
            )
            return None

        # Check if index name is missing and if it's the drop_time param set
        if feature_df.index.name is None:
            logger.warning(
                f"Feature DataFrame for {feature_name} has missing index name with drop_time param set. Skipping as expected."
            )
            return None

        # Ensure index has name "Time" for all other cases
        if feature_df.index.name != "Time":
            feature_df.index.name = "Time"

        rename_map = {}
        columns_to_keep = []  # We no longer need to keep a "Time" column

        for sub_feature in sub_feature_names:
            if sub_feature in feature_df.columns:
                # Use human-readable param string instead of hash for column names
                new_name = f"{sub_feature}"
                rename_map[sub_feature] = new_name
                columns_to_keep.append(sub_feature)  # Keep original name for selection
            else:
                logger.warning(
                    f"Sub-feature '{sub_feature}' not found in computed/cached df for {feature_name} (hash: {param_hash})."
                )

        # Select only the sub-features to be renamed
        feature_df_selected = feature_df[columns_to_keep]

        # Rename the selected columns
        feature_df_renamed = feature_df_selected.rename(columns=rename_map)

        # Final check for DatetimeIndex after operations
        if (
            not isinstance(feature_df_renamed.index, pd.DatetimeIndex)
            or feature_df_renamed.index.name != "Time"
        ):
            logger.critical(
                f"Lost DatetimeIndex during processing for {feature_name} (hash: {param_hash}). This should not happen."
            )
            return None

        return feature_df_renamed

    def compute(self, asset_data: AssetPriceDataSet, symbol: str) -> List[Delayed]:
        """
        Generates a list of delayed tasks for computing or retrieving features.

        Each task corresponds to one feature definition and one parameter set.
        The task, when executed, will return a DataFrame with DatetimeIndex named "Time"
        and the renamed feature columns, or None if skipped/failed.

        Args:
            asset_data: Dataset containing asset price information.
            symbol: The trading symbol being processed.

        Returns:
            List[dask.delayed]: A list of delayed objects ready for computation.
        """
        delayed_tasks = []

        # Use the utility function to prepare the original DataFrame with DatetimeIndex
        original_df = ensure_datetime_index(
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
