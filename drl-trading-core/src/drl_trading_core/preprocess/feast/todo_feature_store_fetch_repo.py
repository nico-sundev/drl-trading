import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd
from drl_trading_common.config.feature_config import FeatureStoreConfig
from feast import Entity, FeatureStore
from injector import inject
from pandas import DataFrame

from drl_trading_core.common.model.asset_price_dataset import AssetPriceDataSet

logger = logging.getLogger(__name__)


class FeatureStoreFetchRepoInterface(ABC):
    """
    Interface for interacting with a feature store.

    This interface defines the contract for services that handle feature storage,
    retrieval, and feature view management.
    """

    @abstractmethod
    def get_entity_value(self, symbol: str, timeframe: str) -> str:
        """
        Get the entity value for the given symbol and timeframe.

        Args:
            symbol: The trading symbol
            timeframe: The timeframe of the data

        Returns:
            str: A unique identifier combining symbol and timeframe
        """
        pass

    @abstractmethod
    def get_feature_view_name(
        self, feature_name: str, param_hash: str, timeframe: str
    ) -> str:
        """
        Create a unique feature view name incorporating timeframe.

        Args:
            feature_name: Name of the feature
            param_hash: Hash of the feature parameters
            timeframe: The timeframe of the data

        Returns:
            str: A unique name for the feature view including timeframe information
        """
        pass

    @abstractmethod
    def get_historical_features(
        self,
        feature_name: str,
        param_hash: str,
        sub_feature_names: List[str],
        asset_data: AssetPriceDataSet,
        symbol: str,
    ) -> Optional[DataFrame]:
        """
        Retrieve historical features from the feature store if available.

        Args:
            feature_name: Name of the feature
            param_hash: Hash of the feature parameters
            sub_feature_names: List of sub-feature names to retrieve
            asset_data: The asset price dataset to match timestamps with
            symbol: The trading symbol

        Returns:
            Optional[DataFrame]: DataFrame containing the retrieved features or None if not found
        """
        pass

    @abstractmethod
    def is_enabled(self) -> bool:
        """
        Check if the feature store is enabled.

        Returns:
            bool: True if the feature store is enabled, False otherwise
        """
        pass

    @abstractmethod
    def get_online_features(
        self,
        entity_df: DataFrame,
        feature_refs: List[str],
    ) -> Optional[DataFrame]:
        """
        Retrieve features from the online feature store.

        Args:
            entity_df: DataFrame containing entity information with timestamps
            feature_refs: List of feature references in format "feature_view:feature_name"

        Returns:
            Optional[DataFrame]: DataFrame containing the retrieved features or None if not available
        """
        pass

    @abstractmethod
    def get_batch_historical_features(
        self,
        feature_requests: List[dict],
        asset_data: AssetPriceDataSet,
        symbol: str,
    ) -> Optional[DataFrame]:
        """
        Retrieve multiple features in a single batch request for optimal performance.

        Args:
            feature_requests: List of feature request dictionaries containing
                            feature_name, param_hash, and sub_feature_names
            asset_data: The asset price dataset to match timestamps with
            symbol: The trading symbol

        Returns:
            Optional[DataFrame]: DataFrame containing all retrieved features or None if not found
        """
        pass


class FeatureStoreFetchRepo(FeatureStoreFetchRepoInterface):
    """
    Service for interacting with the Feast feature store.

    This class handles all interactions with the Feast feature store, including
    initializing the store, creating entities and feature views, storing computed
    features, and retrieving historical features.
    """

    @inject
    def __init__(
        self, config: FeatureStoreConfig, feature_store: Optional[FeatureStore] = None
    ) -> None:
        """
        Initialize the FeastService with configuration.

        Args:
            feature_store_config: Configuration for the feature store
        """
        self.config = config
        self.feature_store = feature_store
        # Add feature view cache for performance
        self._feature_view_cache: Dict[str, bool] = {}

    def _get_entity(self, symbol: str, timeframe: str) -> Entity:
        """
        Create an entity for the given symbol and timeframe.

        This is a private helper method indicated by the underscore prefix.

        Args:
            symbol: The trading symbol
            timeframe: The timeframe of the data

        Returns:
            Entity: Feast entity for this symbol/timeframe combination
        """
        return Entity(
            name=self.config.entity_name,
            join_keys=["symbol", "timeframe"],
            description=f"Entity for {symbol}/{timeframe} asset price data",
        )

    def get_entity_value(self, symbol: str, timeframe: str) -> str:
        """
        Get the entity value for the given symbol and timeframe.

        Args:
            symbol: The trading symbol
            timeframe: The timeframe of the data

        Returns:
            str: A unique identifier combining symbol and timeframe
        """
        return f"{symbol}_{timeframe}"

    def get_feature_view_name(
        self, feature_name: str, param_hash: str, timeframe: str
    ) -> str:
        """
        Create a unique feature view name incorporating timeframe.

        Args:
            feature_name: Name of the feature
            param_hash: Hash of the feature parameters
            timeframe: The timeframe of the data

        Returns:
            str: A unique name for the feature view including timeframe information
        """
        return f"{feature_name}_{timeframe}_{param_hash}"

    def _validate_feature_view_exists(self, feature_view_name: str) -> bool:
        """
        Check if a feature view exists in the store with caching.

        Args:
            feature_view_name: Name of the feature view to check

        Returns:
            bool: True if feature view exists, False otherwise
        """
        if feature_view_name in self._feature_view_cache:
            return self._feature_view_cache[feature_view_name]

        if not self.feature_store:
            self._feature_view_cache[feature_view_name] = False
            return False

        try:
            self.feature_store.get_feature_view(feature_view_name)
            self._feature_view_cache[feature_view_name] = True
            return True
        except Exception:
            self._feature_view_cache[feature_view_name] = False
            return False

    def get_historical_features(
        self,
        feature_name: str,
        param_hash: str,
        sub_feature_names: List[str],
        asset_data: AssetPriceDataSet,
        symbol: str,
    ) -> Optional[DataFrame]:
        """
        Retrieve historical features from the feature store if available.

        Args:
            feature_name: Name of the feature
            param_hash: Hash of the feature parameters
            sub_feature_names: List of sub-feature names to retrieve
            asset_data: The asset price dataset to match timestamps with
            symbol: The trading symbol

        Returns:
            Optional[DataFrame]: DataFrame containing the retrieved features or None if not found
        """
        if not self.feature_store or not self.config.enabled:
            logger.debug("Feature store not enabled or available")
            return None

        # Early validation
        if not sub_feature_names:
            logger.warning(f"No sub-feature names provided for {feature_name}")
            return None

        if asset_data.asset_price_dataset.empty:
            logger.warning(f"Empty asset data provided for {feature_name}")
            return None

        feature_view_name = self.get_feature_view_name(
            feature_name, param_hash, asset_data.timeframe
        )

        # Check if feature view exists before attempting retrieval
        if not self._validate_feature_view_exists(feature_view_name):
            logger.debug(f"Feature view {feature_view_name} does not exist")
            return None

        try:
            # Optimize entity DataFrame creation
            entity_df = self._create_optimized_entity_df(asset_data, symbol)

            # Create feature references
            feature_refs = [f"{feature_view_name}:{name}" for name in sub_feature_names]

            logger.debug(f"Attempting to retrieve features from store: {feature_refs}")

            # Use Feast's get_historical_features with optimized parameters
            retrieval_job = self.feature_store.get_historical_features(
                entity_df=entity_df,
                features=feature_refs,
                full_feature_names=True  # Get fully qualified feature names
            )

            historical_features = retrieval_job.to_df()

            if historical_features.empty:
                logger.debug(f"No historical features found for {feature_view_name}")
                return None

            # Validate and optimize the returned DataFrame
            validated_features = self._validate_and_optimize_features(
                historical_features, sub_feature_names, feature_view_name
            )

            if validated_features is not None:
                logger.info(f"Successfully retrieved {len(validated_features)} feature records for {feature_view_name}")
                return validated_features

        except Exception as e:
            logger.warning(f"Failed to retrieve features from store for {feature_view_name}: {str(e)}")

        return None

    def _create_optimized_entity_df(self, asset_data: AssetPriceDataSet, symbol: str) -> DataFrame:
        """
        Create an optimized entity DataFrame for feature retrieval.

        Args:
            asset_data: The asset price dataset
            symbol: The trading symbol

        Returns:
            DataFrame: Optimized entity DataFrame
        """
        # Use vectorized operations for better performance
        time_data = asset_data.asset_price_dataset["Time"]

        entity_df = DataFrame({
            "event_timestamp": pd.to_datetime(time_data),
            self.config.entity_name: self.get_entity_value(symbol, asset_data.timeframe)
        })

        # Ensure proper sorting for optimal Feast performance
        entity_df = entity_df.sort_values("event_timestamp").reset_index(drop=True)

        return entity_df

    def _validate_and_optimize_features(
        self,
        features_df: DataFrame,
        expected_sub_features: List[str],
        feature_view_name: str
    ) -> Optional[DataFrame]:
        """
        Validate and optimize the retrieved features DataFrame.

        Args:
            features_df: Raw features DataFrame from Feast
            expected_sub_features: List of expected sub-feature names
            feature_view_name: Name of the feature view for logging

        Returns:
            Optional[DataFrame]: Validated and optimized DataFrame or None if validation fails
        """
        if features_df.empty:
            return None

        # Check for required columns
        expected_columns = [f"{feature_view_name}__{name}" for name in expected_sub_features]
        missing_columns = [col for col in expected_columns if col not in features_df.columns]

        if missing_columns:
            logger.warning(f"Missing expected columns in {feature_view_name}: {missing_columns}")
            # Try without feature view prefix (fallback)
            expected_columns = expected_sub_features
            missing_columns = [col for col in expected_columns if col not in features_df.columns]

            if missing_columns:
                logger.error(f"Critical columns missing in {feature_view_name}: {missing_columns}")
                return None

        # Remove Feast metadata columns and optimize data types
        metadata_columns = [self.config.entity_name, "event_timestamp"]
        feature_columns = [col for col in features_df.columns if col not in metadata_columns]

        # Create optimized result DataFrame
        result_df = features_df[feature_columns].copy()

        # Add time index if event_timestamp is available
        if "event_timestamp" in features_df.columns:
            result_df["Time"] = features_df["event_timestamp"]
            result_df = result_df.set_index("Time")

        # Optimize data types for memory efficiency
        for col in result_df.select_dtypes(include=['float64']).columns:
            result_df[col] = result_df[col].astype('float32')

        return result_df

    def get_batch_historical_features(
        self,
        feature_requests: List[dict],
        asset_data: AssetPriceDataSet,
        symbol: str,
    ) -> Optional[DataFrame]:
        """
        Retrieve multiple features in a single batch request for optimal performance.

        This method optimizes feature retrieval when multiple features are needed
        by combining them into a single Feast query.

        Args:
            feature_requests: List of feature request dictionaries containing
                            feature_name, param_hash, and sub_feature_names
            asset_data: The asset price dataset to match timestamps with
            symbol: The trading symbol

        Returns:
            Optional[DataFrame]: DataFrame containing all retrieved features or None if not found
        """
        if not self.feature_store or not self.config.enabled or not feature_requests:
            return None

        try:
            # Create entity DataFrame once for all features
            entity_df = self._create_optimized_entity_df(asset_data, symbol)

            # Collect all feature references
            all_feature_refs = []
            feature_view_names = []

            for request in feature_requests:
                feature_view_name = self.get_feature_view_name(
                    request["feature_name"],
                    request["param_hash"],
                    asset_data.timeframe
                )

                # Only include if feature view exists
                if self._validate_feature_view_exists(feature_view_name):
                    feature_refs = [
                        f"{feature_view_name}:{name}"
                        for name in request["sub_feature_names"]
                    ]
                    all_feature_refs.extend(feature_refs)
                    feature_view_names.append(feature_view_name)

            if not all_feature_refs:
                logger.debug("No valid feature views found for batch request")
                return None

            logger.debug(f"Batch retrieving features: {all_feature_refs}")

            # Single Feast query for all features
            retrieval_job = self.feature_store.get_historical_features(
                entity_df=entity_df,
                features=all_feature_refs,
                full_feature_names=True
            )

            batch_features = retrieval_job.to_df()

            if not batch_features.empty:
                logger.info(f"Successfully retrieved batch features for {len(feature_view_names)} feature views")
                return self._validate_and_optimize_features(
                    batch_features,
                    [], # Skip individual validation for batch
                    "batch_request"
                )

        except Exception as e:
            logger.warning(f"Failed to retrieve batch features: {str(e)}")

        return None

    def is_enabled(self) -> bool:
        """
        Check if the feature store is enabled.

        Returns:
            bool: True if the feature store is enabled, False otherwise
        """
        return self.config.enabled and self.feature_store is not None

    def get_online_features(
        self,
        entity_df: DataFrame,
        feature_refs: List[str],
    ) -> Optional[DataFrame]:
        """
        Retrieve features from the online feature store.

        This method leverages Feast's online serving capabilities to retrieve
        pre-computed features for real-time inference.

        Args:
            entity_df: DataFrame containing entity information with timestamps
            feature_refs: List of feature references in format "feature_view:feature_name"

        Returns:
            Optional[DataFrame]: DataFrame containing the retrieved features or None if not available
        """
        if (
            not self.feature_store
            or not self.config.enabled
            or not self.config.online_enabled
        ):
            logger.debug("Online feature serving not enabled or available")
            return None

        try:
            logger.debug(f"Retrieving online features: {feature_refs}")

            # Use Feast's get_online_features method
            online_response = self.feature_store.get_online_features(
                entity_rows=entity_df.to_dict("records"), features=feature_refs
            )

            # Convert response to DataFrame
            online_df = online_response.to_df()

            if online_df.empty:
                logger.debug("No online features retrieved")
                return None

            logger.debug(
                f"Successfully retrieved {len(online_df)} online feature records"
            )
            return online_df

        except Exception as e:
            logger.warning(f"Failed to retrieve online features: {str(e)}")
            return None

    def clear_cache(self) -> None:
        """Clear the feature view cache."""
        self._feature_view_cache.clear()
        logger.debug("Feature view cache cleared")
