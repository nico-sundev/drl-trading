import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd
from dask import delayed
from dask.delayed import Delayed
from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.config.feature_config import FeaturesConfig
from injector import inject
from pandas import DataFrame

from drl_trading_core.common.model.asset_price_dataset import AssetPriceDataSet
from drl_trading_core.preprocess.feast.todo_feature_store_fetch_repo import (
    FeatureStoreFetchRepoInterface,
)
from drl_trading_core.preprocess.feature.feature_manager import (
    FeatureManager,
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
    def fetch_features_batch(
        self, asset_data: AssetPriceDataSet, symbol: str
    ) -> List[Delayed]:
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

    @abstractmethod
    def fetch_features_incremental(
        self, asset_data: AssetPriceDataSet, symbol: str
    ) -> List[Delayed]:
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

    @abstractmethod
    def fetch_features_batch_optimized(
        self, asset_data: AssetPriceDataSet, symbol: str
    ) -> List[Delayed]:
        """
        Optimized batch feature retrieval using single Feast query for multiple features.

        This method leverages the get_batch_historical_features capability to retrieve
        multiple features in a single query, reducing the number of round trips to Feast.

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
        feast_service: FeatureStoreFetchRepoInterface,
        feature_manager_service: FeatureManager,
    ) -> None:
        """
        Initialize the FeatureAggregator with configuration and services.

        Args:
            config: Configuration for feature definitions.
            feast_service: Service for interacting with the Feast feature store.
            computing_service: Service for computing features.
        """
        self.config = config
        self.feast_service = feast_service
        self.feature_manager_service = feature_manager_service

    def _get_single_feature_offline(
        self,
        feature: BaseFeature,
        symbol: str,
        asset_data: AssetPriceDataSet,
    ) -> Optional[DataFrame]:
        """
        Retrieves a single feature for a given parameter set from the feature store,
        or delegates computation to the ComputingService if not found in store.
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

        feature_name = feature.get_feature_name()
        param_hash = feature.get_config().hash_id()
        sub_feature_names = feature.get_sub_features_names()

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
        if historical_features is None or historical_features.empty:
            logger.debug(
                f"No cached features found for {feature_name} with params hash {param_hash}"
            )
            return None

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

        logger.debug(
            f"Using cached features for {feature_name} with params hash {param_hash}"
        )
        feature_df = historical_features

        if feature_df is None or feature_df.empty:
            logger.warning(
                f"No features generated or retrieved for {feature_name} with params hash {param_hash}"
            )
            return None

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

        return feature_df_renamed

    def _get_single_feature_online(
        self,
        feature: BaseFeature,
        symbol: str,
        asset_data: AssetPriceDataSet,
    ) -> Optional[DataFrame]:
        """
        Retrieves a single feature for a given parameter set from the feature store,
        or delegates computation to the ComputingService if not found in store.
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

        feature_name = feature.get_feature_name()
        param_hash = feature.get_config().hash_id()
        sub_feature_names = feature.get_sub_features_names()

        # Try to get features from store first
        historical_features: Optional[DataFrame] = None
        if self.feast_service.is_enabled():
            historical_features = self.feast_service.get_online_features(
                # feature_name=feature_name,
                # param_hash=param_hash,
                # sub_feature_names=sub_feature_names,
                # asset_data=asset_data,
                # symbol=symbol,
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

        if feature_df is None or feature_df.empty:
            logger.warning(
                f"No features generated or retrieved for {feature_name} with params hash {param_hash}"
            )
            return None

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

        return feature_df_renamed

    def fetch_features_batch(
        self, asset_data: AssetPriceDataSet, symbol: str
    ) -> List[Delayed]:
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

        for feature in self.feature_manager_service.get_all_features():
            # Create a delayed task for each feature/param combination
            task = delayed(self._get_single_feature_offline)(
                feature, symbol, asset_data
            )
            delayed_tasks.append(task)

        logger.info(
            f"Generated {len(delayed_tasks)} delayed feature computation tasks for symbol {symbol}."
        )
        return delayed_tasks

    def fetch_features_incremental(
        self, asset_data: AssetPriceDataSet, symbol: str
    ) -> List[Delayed]:
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

        for feature in self.feature_manager_service.get_all_features():
            # Create a delayed task for each feature/param combination
            task = delayed(self._get_single_feature_online)(
                feature, symbol, asset_data
            )
            delayed_tasks.append(task)

        logger.info(
            f"Generated {len(delayed_tasks)} delayed feature computation tasks for symbol {symbol}."
        )
        return delayed_tasks

    def fetch_features_batch_optimized(
        self, asset_data: AssetPriceDataSet, symbol: str
    ) -> List[Delayed]:
        """
        Optimized batch feature retrieval using single Feast query for multiple features.

        This method leverages the get_batch_historical_features capability to retrieve
        multiple features in a single query, reducing the number of round trips to Feast.

        Args:
            asset_data: Dataset containing asset price information.
            symbol: The trading symbol being processed.

        Returns:
            List[dask.delayed]: A list of delayed objects ready for computation.
        """
        if not self.feast_service.is_enabled():
            logger.warning("Feast service is not enabled, falling back to standard batch processing")
            return self.fetch_features_batch(asset_data, symbol)

        # Prepare batch request for all features
        feature_requests = []
        for feature in self.feature_manager_service.get_all_features():
            feature_requests.append({
                'feature_name': feature.get_feature_name(),
                'param_hash': feature.get_config().hash_id(),
                'sub_feature_names': feature.get_sub_features_names()
            })

        if not feature_requests:
            logger.info(f"No features to retrieve for symbol {symbol}")
            return []

        # Create a single delayed task for batch retrieval
        def _get_batch_features_optimized():
            try:
                batch_results = self.feast_service.get_batch_historical_features(
                    feature_requests=feature_requests,
                    asset_data=asset_data,
                    symbol=symbol
                )

                if not batch_results:
                    logger.warning(f"No batch features retrieved for symbol {symbol}")
                    return []

                # Process and split results back into individual feature DataFrames
                processed_features = []
                for i, feature_request in enumerate(feature_requests):
                    if i < len(batch_results) and batch_results[i] is not None:
                        feature_df = batch_results[i]

                        # Ensure proper time index
                        if not isinstance(feature_df.index, pd.DatetimeIndex):
                            if "Time" in feature_df.columns:
                                try:
                                    feature_df["Time"] = pd.to_datetime(feature_df["Time"])
                                    feature_df = feature_df.set_index("Time")
                                except Exception:
                                    logger.warning(f"Invalid 'Time' column in batch result {i}")
                                    continue
                            else:
                                logger.warning(f"Missing DatetimeIndex in batch result {i}")
                                continue

                        # Apply column renaming for sub-features
                        rename_map = {}
                        columns_to_keep = []

                        for sub_feature in feature_request['sub_feature_names']:
                            if sub_feature in feature_df.columns:
                                rename_map[sub_feature] = sub_feature
                                columns_to_keep.append(sub_feature)
                            else:
                                logger.warning(
                                    f"Sub-feature '{sub_feature}' not found in batch result for "
                                    f"{feature_request['feature_name']} (hash: {feature_request['param_hash']})"
                                )

                        if columns_to_keep:
                            feature_df_selected = feature_df[columns_to_keep]
                            feature_df_renamed = feature_df_selected.rename(columns=rename_map)
                            processed_features.append(feature_df_renamed)
                        else:
                            processed_features.append(None)
                    else:
                        processed_features.append(None)

                return processed_features

            except Exception as e:
                logger.error(f"Error in batch feature retrieval for symbol {symbol}: {e}")
                return []

        # Create delayed task for batch processing
        batch_task = delayed(_get_batch_features_optimized)()

        logger.info(f"Generated optimized batch feature retrieval task for {len(feature_requests)} features for symbol {symbol}")
        return [batch_task]
