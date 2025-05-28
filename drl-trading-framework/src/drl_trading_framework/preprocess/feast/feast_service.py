import logging
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import List, Optional

from drl_trading_common.config.feature_config import FeatureStoreConfig
from feast import Entity, FeatureStore, FeatureView, Field, FileSource
from feast.types import Float32
from injector import inject
from pandas import DataFrame

from drl_trading_framework.common.model.asset_price_dataset import AssetPriceDataSet

logger = logging.getLogger(__name__)


class FeastServiceInterface(ABC):
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
    def store_computed_features(
        self,
        feature_df: DataFrame,
        feature_name: str,
        param_hash: str,
        sub_feature_names: List[str],
        asset_data: AssetPriceDataSet,
        symbol: str,
    ) -> None:
        """
        Store computed features in the feature store.

        Args:
            feature_df: DataFrame containing the computed features
            feature_name: Name of the feature
            param_hash: Hash of the feature parameters
            sub_feature_names: List of sub-feature names in the feature
            asset_data: The asset price dataset containing metadata
            symbol: The trading symbol
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


class FeastService(FeastServiceInterface):
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
            join_keys=[self.config.entity_name],
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
            return None

        try:
            # Create entity DataFrame for feature retrieval
            entity_df = DataFrame()
            entity_df["Time"] = asset_data.asset_price_dataset["Time"]
            entity_df["event_timestamp"] = asset_data.asset_price_dataset["Time"]
            entity_df[self.config.entity_name] = self.get_entity_value(
                symbol, asset_data.timeframe
            )

            # Get feature view name based on feature, timeframe and params
            feature_view_name = self.get_feature_view_name(
                feature_name, param_hash, asset_data.timeframe
            )
            feature_refs = [f"{feature_view_name}:{name}" for name in sub_feature_names]

            logger.info(f"Attempting to retrieve features from store: {feature_refs}")
            historical_features = self.feature_store.get_historical_features(
                entity_df=entity_df, features=feature_refs
            ).to_df()

            if not historical_features.empty:
                logger.info(f"Successfully retrieved features for {feature_view_name}")
                return historical_features

        except Exception as e:
            logger.warning(f"Failed to retrieve features from store: {str(e)}")

        return None

    def store_computed_features(
        self,
        feature_df: DataFrame,
        feature_name: str,
        param_hash: str,
        sub_feature_names: List[str],
        asset_data: AssetPriceDataSet,
        symbol: str,
    ) -> None:
        """
        Store computed features in the feature store.

        Args:
            feature_df: DataFrame containing the computed features
            feature_name: Name of the feature
            param_hash: Hash of the feature parameters
            sub_feature_names: List of sub-feature names in the feature
            asset_data: The asset price dataset containing metadata
            symbol: The trading symbol
        """
        if not self.feature_store or not self.config.enabled:
            return

        try:
            # Make a copy to avoid modifying the original dataframe
            store_df = feature_df.copy()

            # Add required columns for feast
            store_df["event_timestamp"] = store_df["Time"]
            store_df[self.config.entity_name] = self.get_entity_value(
                symbol, asset_data.timeframe
            )

            feature_view_name = self.get_feature_view_name(
                feature_name, param_hash, asset_data.timeframe
            )

            # Create file path based on feature name, symbol and timeframe
            file_path = (
                f"{self.config.offline_store_path}/"
                f"{symbol}_{asset_data.timeframe}_{feature_name}_{param_hash}.parquet"
            )

            logger.info(
                f"Storing computed features in feature store: {feature_view_name} at {file_path}"
            )

            # Save features to parquet file
            store_df.to_parquet(
                file_path,
                index=False,
            )

            # Check if feature view exists
            feature_view_exists = True
            try:
                self.feature_store.get_feature_view(feature_view_name)
            except Exception:
                feature_view_exists = False
                logger.info(
                    f"Feature view {feature_view_name} does not exist, creating..."
                )

            # Create and apply feature view if needed
            if not feature_view_exists:
                # Create entity for this symbol and timeframe
                entity = self._get_entity(symbol, asset_data.timeframe)

                feature_view = self._create_feature_view(
                    feature_name=feature_name,
                    param_hash=param_hash,
                    sub_feature_names=sub_feature_names,
                    source_path=file_path,
                    entity=entity,
                    symbol=symbol,
                    timeframe=asset_data.timeframe,
                )

                self.feature_store.apply([entity, feature_view])

            # Materialize the feature view to make it available for online serving
            if self.config.online_enabled:
                self.feature_store.materialize(
                    start_date=store_df["event_timestamp"].min(),
                    end_date=store_df["event_timestamp"].max(),
                )

        except Exception as e:
            logger.warning(f"Failed to store features: {str(e)}")

    def _create_feature_view(
        self,
        feature_name: str,
        param_hash: str,
        sub_feature_names: List[str],
        source_path: str,
        entity: Entity,
        symbol: str,
        timeframe: str,
    ) -> FeatureView:
        """
        Create a feature view for the given feature parameters.

        Args:
            feature_name: Name of the feature
            param_hash: Hash of the feature parameters
            sub_feature_names: List of sub-feature names in the feature
            source_path: Path to the source data file
            entity: The entity to associate with this feature view
            symbol: The trading symbol
            timeframe: The timeframe of the data

        Returns:
            FeatureView: The created feature view
        """
        feature_view_name = self.get_feature_view_name(
            feature_name, param_hash, timeframe
        )

        # Create a file source for the feature
        source = FileSource(
            name=f"{feature_view_name}_source",
            path=source_path,
            timestamp_field="event_timestamp",
        )

        # Create fields for the feature view
        fields = []
        for sub_feature in sub_feature_names:
            fields.append(Field(name=sub_feature, dtype=Float32))

        # Create and return the feature view
        return FeatureView(
            name=feature_view_name,
            entities=[entity],
            ttl=timedelta(days=self.config.ttl_days),
            schema=fields,
            online=self.config.online_enabled,
            source=source,
            tags={
                "symbol": symbol,
                "timeframe": timeframe,
            },
        )

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
