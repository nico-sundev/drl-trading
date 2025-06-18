import logging
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import List, Optional

from drl_trading_common.config.feature_config import FeatureStoreConfig
from feast import Entity, FeatureStore, FeatureView, Field, FileSource
from feast.types import Float32
from injector import inject
from pandas import DataFrame

from drl_trading_core.common.model.asset_price_dataset import AssetPriceDataSet

logger = logging.getLogger(__name__)

class FeatureStoreSaveRepoInterface(ABC):

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

        :return: True if the feature store is enabled, False otherwise.
        """
        pass


@inject
class FeatureStoreSaveRepo(FeatureStoreSaveRepoInterface):
    """
    Repository for saving features to a feature store.
    """

    def __init__(self, config: FeatureStoreConfig, feature_store: Optional[FeatureStore] = None):
        self.config = config
        self.feature_store = feature_store

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

    def is_enabled(self) -> bool:
        """
        Check if the feature store is enabled.

        Returns:
            bool: True if the feature store is enabled, False otherwise
        """
        return self.config.enabled and self.feature_store is not None
