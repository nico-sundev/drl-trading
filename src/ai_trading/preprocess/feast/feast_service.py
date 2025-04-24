import logging
from datetime import timedelta
from typing import List, Optional

from feast import Entity, FeatureStore, FeatureView, Field, FileSource
from feast.types import Float32
from pandas import DataFrame

from ai_trading.config.feature_config import FeatureStoreConfig
from ai_trading.model.asset_price_dataset import AssetPriceDataSet

logger = logging.getLogger(__name__)


class FeastService:
    """
    Service for interacting with the Feast feature store.

    This class handles all interactions with the Feast feature store, including
    initializing the store, creating entities and feature views, storing computed
    features, and retrieving historical features.
    """

    def __init__(
        self,
        feature_store_config: FeatureStoreConfig,
        symbol: str,
        asset_data: AssetPriceDataSet,
    ) -> None:
        """
        Initialize the FeastService with configuration and data context.

        Args:
            feature_store_config: Configuration for the feature store
            symbol: The trading symbol being processed
            asset_data: Dataset containing asset price information
        """
        self.config = feature_store_config
        self.symbol = symbol
        self.asset_data = asset_data
        self.feature_store = None
        self.entity = None

        if feature_store_config.enabled:
            logger.info("Initializing feature store connection")
            self.feature_store = FeatureStore(repo_path=feature_store_config.repo_path)

            # Define entity for this symbol and timeframe
            self.entity = Entity(
                name=self.config.entity_name,
                join_keys=[self.config.entity_name],
                description=f"Entity for {self.symbol}/{self.asset_data.timeframe} asset price data",
            )

    def get_entity_value(self) -> str:
        """
        Get the entity value for the current asset data, combining symbol and timeframe.

        Returns:
            str: A unique identifier combining symbol and timeframe
        """
        return f"{self.symbol}_{self.asset_data.timeframe}"

    def get_feature_view_name(self, feature_name: str, param_hash: str) -> str:
        """
        Create a unique feature view name incorporating timeframe.

        Args:
            feature_name: Name of the feature
            param_hash: Hash of the feature parameters

        Returns:
            str: A unique name for the feature view including timeframe information
        """
        return f"{feature_name}_{self.asset_data.timeframe}_{param_hash}"

    def get_historical_features(
        self, feature_name: str, param_hash: str, sub_feature_names: List[str]
    ) -> Optional[DataFrame]:
        """
        Retrieve historical features from the feature store if available.

        Args:
            feature_name: Name of the feature
            param_hash: Hash of the feature parameters
            sub_feature_names: List of sub-feature names to retrieve

        Returns:
            Optional[DataFrame]: DataFrame containing the retrieved features or None if not found
        """
        if not self.feature_store or not self.config.enabled:
            return None

        try:
            # Create entity DataFrame for feature retrieval
            entity_df = DataFrame()
            entity_df["Time"] = self.asset_data.asset_price_dataset["Time"]
            entity_df["event_timestamp"] = self.asset_data.asset_price_dataset["Time"]
            entity_df[self.config.entity_name] = self.get_entity_value()

            # Get feature view name based on feature, timeframe and params
            feature_view_name = self.get_feature_view_name(feature_name, param_hash)
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
    ) -> None:
        """
        Store computed features in the feature store.

        Args:
            feature_df: DataFrame containing the computed features
            feature_name: Name of the feature
            param_hash: Hash of the feature parameters
            sub_feature_names: List of sub-feature names in the feature
        """
        if not self.feature_store or not self.config.enabled:
            return

        try:
            # Make a copy to avoid modifying the original dataframe
            store_df = feature_df.copy()

            # Add required columns for feast
            store_df["event_timestamp"] = store_df["Time"]
            store_df[self.config.entity_name] = self.get_entity_value()

            feature_view_name = self.get_feature_view_name(feature_name, param_hash)

            # Create file path based on feature name, symbol and timeframe
            file_path = (
                f"{self.config.offline_store_path}/"
                f"{self.symbol}_{self.asset_data.timeframe}_{feature_name}_{param_hash}.parquet"
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
                feature_view = self._create_feature_view(
                    feature_name=feature_name,
                    param_hash=param_hash,
                    sub_feature_names=sub_feature_names,
                    source_path=file_path,
                )

                if self.entity is not None:
                    self.feature_store.apply([self.entity, feature_view])
                else:
                    logger.warning("Entity is None, skipping feature view application.")

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
    ) -> FeatureView:
        """
        Create a feature view for the given feature parameters.

        Args:
            feature_name: Name of the feature
            param_hash: Hash of the feature parameters
            sub_feature_names: List of sub-feature names in the feature
            source_path: Path to the source data file

        Returns:
            FeatureView: The created feature view
        """
        feature_view_name = self.get_feature_view_name(feature_name, param_hash)

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
            entities=[self.entity] if self.entity is not None else [],
            ttl=timedelta(days=self.config.ttl_days),
            schema=fields,
            online=self.config.online_enabled,
            source=source,
            tags={
                "symbol": self.symbol,
                "timeframe": self.asset_data.timeframe,
            },
        )

    def is_enabled(self) -> bool:
        """
        Check if the feature store is enabled.

        Returns:
            bool: True if the feature store is enabled, False otherwise
        """
        return self.config.enabled and self.feature_store is not None
