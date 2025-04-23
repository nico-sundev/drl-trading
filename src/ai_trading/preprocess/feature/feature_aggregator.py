import logging
from datetime import timedelta
from typing import Optional

from feast import Entity, FeatureStore, FeatureView, Field, FileSource
from feast.types import Float32
from pandas import DataFrame, concat

from ai_trading.config.feature_config import FeaturesConfig, FeatureStoreConfig
from ai_trading.model.asset_price_dataset import AssetPriceDataSet
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature
from ai_trading.preprocess.feature.feature_class_registry import FeatureClassRegistry

logger = logging.getLogger(__name__)


class FeatureAggregator:
    def __init__(
        self,
        asset_data: AssetPriceDataSet,
        symbol: str,
        config: FeaturesConfig,
        class_registry: FeatureClassRegistry,
        feature_store_config: FeatureStoreConfig,
    ) -> None:
        self.asset_data = asset_data
        self.config = config
        self.symbol = symbol
        self.class_registry = class_registry
        self.feature_store_config = feature_store_config
        self.feature_store = None
        self.entity = None

        if feature_store_config.enabled:
            logger.info("Initializing feature store connection")
            self.feature_store = FeatureStore(repo_path=feature_store_config.repo_path)
            # Define entity for this symbol and timeframe
            self.entity = Entity(
                name=self.feature_store_config.entity_name,
                join_keys=[self.feature_store_config.entity_name],
                description=f"Entity for {self.symbol}/{self.asset_data.timeframe} asset price data",
            )

    def _get_entity_value(self) -> str:
        """
        Get the entity value for the current asset data, combining symbol and timeframe.

        Returns:
            str: A unique identifier combining symbol and timeframe
        """
        return f"{self.symbol}_{self.asset_data.timeframe}"

    def _get_feature_view_name(self, feature_name: str, param_hash: str) -> str:
        """
        Create a unique feature view name incorporating timeframe.

        Args:
            feature_name: Name of the feature
            param_hash: Hash of the feature parameters

        Returns:
            str: A unique name for the feature view including timeframe information
        """
        return f"{feature_name}_{self.asset_data.timeframe}_{param_hash}"

    def _get_historical_features(self, feature: BaseFeature) -> Optional[DataFrame]:
        """Try to retrieve features from feature store if available."""
        if not self.feature_store:
            return None

        try:
            # Create entity DataFrame for feature retrieval
            entity_df = DataFrame()
            entity_df["Time"] = self.asset_data.asset_price_dataset["Time"]
            entity_df["event_timestamp"] = self.asset_data.asset_price_dataset["Time"]
            entity_df[self.feature_store_config.entity_name] = self._get_entity_value()

            # Get feature view name based on feature, timeframe and params
            feature_view_name = self._get_feature_view_name(
                feature_name=feature.get_feature_name(),
                param_hash=feature.config.hash_id(),
            )
            feature_refs = [
                f"{feature_view_name}:{name}"
                for name in feature.get_sub_features_names()
            ]

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

    def compute(self) -> DataFrame:
        feature_results = []
        for feature in self.config.feature_definitions:
            if not feature.enabled:
                continue

            for param_set in feature.parsed_parameter_sets:
                if not param_set.enabled:
                    continue

                feature_class = self.class_registry.feature_class_map[feature.name]
                feature_instance = feature_class(
                    source=self.asset_data.asset_price_dataset, config=param_set
                )

                # Try to get features from store first
                historical_features = self._get_historical_features(feature_instance)
                if historical_features is not None:
                    feature_results.append(historical_features)
                    continue

                # Compute features if not found in store
                logger.info(
                    f"Computing features for {feature.name} with params {param_set}"
                )
                feature_df = feature_instance.compute()

                # Store computed features if feature store is enabled
                if self.feature_store:
                    self._store_computed_features(feature_df, feature_instance)

                feature_results.append(feature_df)

        if not feature_results:
            return DataFrame()

        return concat(
            [df.set_index("Time") for df in feature_results], axis=1
        ).reset_index()

    def _create_feature_view(
        self, feature_instance: BaseFeature, source_path: str
    ) -> FeatureView:
        """Create a feature view for the given feature."""
        feature_name = feature_instance.get_feature_name()
        param_hash = feature_instance.config.hash_id()
        feature_view_name = self._get_feature_view_name(feature_name, param_hash)

        # Create a file source for the feature
        source = FileSource(
            name=f"{feature_view_name}_source",
            path=source_path,
            timestamp_field="event_timestamp",
        )

        # Create fields for the feature view
        fields = []
        for sub_feature in feature_instance.get_sub_features_names():
            fields.append(Field(name=sub_feature, dtype=Float32))

        # Create and return the feature view
        return FeatureView(
            name=feature_view_name,
            entities=[self.entity] if self.entity is not None else [],
            ttl=timedelta(days=self.feature_store_config.ttl_days),
            schema=fields,
            online=self.feature_store_config.online_enabled,
            source=source,
            tags={
                "symbol": self.symbol,
                "timeframe": self.asset_data.timeframe,
            },
        )

    def _store_computed_features(
        self,
        feature_df: DataFrame,
        feature_instance: BaseFeature,
    ) -> None:
        """Store computed features in the feature store."""
        if not self.feature_store:
            return

        try:
            # Make a copy to avoid modifying the original dataframe
            store_df = feature_df.copy()

            # Add required columns for feast
            store_df["event_timestamp"] = store_df["Time"]
            store_df[self.feature_store_config.entity_name] = self._get_entity_value()

            feature_name = feature_instance.get_feature_name()
            param_hash = feature_instance.config.hash_id()
            feature_view_name = self._get_feature_view_name(feature_name, param_hash)

            # Create file path based on feature name, symbol and timeframe
            file_path = (
                f"{self.feature_store_config.offline_store_path}/"
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
                    feature_instance=feature_instance, source_path=file_path
                )
                if self.entity is not None:
                    self.feature_store.apply([self.entity, feature_view])
                else:
                    logger.warning("Entity is None, skipping feature view application.")

            # Materialize the feature view to make it available for online serving
            if self.feature_store_config.online_enabled:
                self.feature_store.materialize(
                    start_date=store_df["event_timestamp"].min(),
                    end_date=store_df["event_timestamp"].max(),
                )

        except Exception as e:
            logger.warning(f"Failed to store features: {str(e)}")
