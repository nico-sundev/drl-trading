import logging
from typing import Optional

from feast import FeatureStore
from pandas import DataFrame, concat

from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig
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

        if feature_store_config.enabled:
            logger.info("Initializing feature store connection")
            self.feature_store = FeatureStore(repo_path=feature_store_config.repo_path)

    def _get_historical_features(self, feature: BaseFeature) -> Optional[DataFrame]:
        """Try to retrieve features from feature store if available."""
        if not self.feature_store:
            return None

        try:
            # Create entity DataFrame for feature retrieval
            entity_df = DataFrame()
            entity_df["Time"] = self.asset_data.asset_price_dataset["Time"]
            entity_df["event_timestamp"] = self.asset_data.asset_price_dataset["Time"]
            # TODO
            # entity_df[self.feature_store_config.entity_name] = (
            #     self._get_symbol_from_df()
            # )

            # Get feature view name based on feature and params
            feature_view_name = (
                f"{feature.get_feature_name()}_{feature.config.hash_id()}"
            )
            feature_refs = feature.get_sub_features_names()

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
                    self._store_computed_features(feature_df, feature.name, param_set)

                feature_results.append(feature_df)

        if not feature_results:
            return DataFrame()

        return concat(
            [df.set_index("Time") for df in feature_results], axis=1
        ).reset_index()

    def _store_computed_features(
        self,
        feature_df: DataFrame,
        feature_name: str,
        param_set: BaseParameterSetConfig,
    ) -> None:
        """Store computed features in the feature store."""
        if not self.feature_store:
            return

        try:
            # Add required columns for feast
            feature_df["event_timestamp"] = feature_df["Time"]
            # TODO Check this
            # feature_df[self.feature_store_config.entity_name] = (
            #     self._get_symbol_from_df()
            # )

            feature_view_name = f"{feature_name}_{param_set.hash_id()}"
            logger.info(
                f"Storing computed features in feature store: {feature_view_name}"
            )

            # Save features to parquet file
            feature_df.to_parquet(
                self.feature_store_config.offline_store_path,
                index=False,
                partition_cols=["event_timestamp"],
            )

            # Apply feature view if it doesn't exist
            try:
                self.feature_store.get_feature_view(feature_view_name)
            except Exception:
                logger.info(
                    f"Feature view {feature_view_name} does not exist, creating..."
                )
                from ai_trading.feature_repo.feature_store_service import (
                    FeatureStoreService,
                )

                service = FeatureStoreService(
                    config=self.config,
                    feature_store_config=self.feature_store_config,
                    class_registry=self.class_registry,
                )
                feature_views = service.create_feature_views()
                self.feature_store.apply([service.entity, *feature_views])

            # Materialize the feature view
            self.feature_store.materialize(
                start_date=feature_df["event_timestamp"].min(),
                end_date=feature_df["event_timestamp"].max(),
            )

        except Exception as e:
            logger.warning(f"Failed to store features: {str(e)}")
