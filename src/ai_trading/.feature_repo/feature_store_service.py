import logging
from datetime import timedelta
from typing import List, Type

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32
from pandas import DataFrame

from ai_trading.config.feature_config import FeaturesConfig, FeatureStoreConfig
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature
from ai_trading.preprocess.feature.feature_class_registry import FeatureClassRegistry

logger = logging.getLogger(__name__)


class FeatureStoreService:
    def __init__(
        self,
        config: FeaturesConfig,
        feature_store_config: FeatureStoreConfig,
        class_registry: FeatureClassRegistry,
    ) -> None:
        self.config = config
        self.store_config = feature_store_config
        self.class_registry = class_registry
        self.entity = Entity(
            name=feature_store_config.entity_name,
            join_keys=[feature_store_config.entity_name],
        )

    def create_feature_views(self) -> List[FeatureView]:
        """Dynamically create feature views based on feature definitions."""
        feature_views = []

        # Create the data source
        data_source = FileSource(
            path=self.store_config.offline_store_path, timestamp_field="event_timestamp"
        )

        for feature_def in self.config.feature_definitions:
            if not feature_def.enabled:
                continue

            for param_set in feature_def.parsed_parameter_sets:
                if not param_set.enabled:
                    continue

                logger.info(
                    f"Creating feature view for {feature_def.name} with params {param_set}"
                )
                try:
                    # Get feature class
                    feature_class: Type[BaseFeature] = (
                        self.class_registry.feature_class_map[feature_def.name]
                    )

                    # Create feature instance with empty DataFrame
                    feature_instance = feature_class(DataFrame())
                    feature_names = feature_instance.get_sub_features_names(param_set)

                    # Create feature view
                    feature_view = FeatureView(
                        name=f"{feature_def.name}_{param_set.hash_id}",  # hash_id is now a computed field
                        entities=[self.entity],
                        ttl=timedelta(days=self.store_config.ttl_days),
                        schema=[
                            Field(name=feature_name, dtype=Float32)
                            for feature_name in feature_names
                        ],
                        online=self.store_config.online_enabled,
                        source=data_source,
                        tags={
                            "feature_type": feature_def.name,
                            "parameter_set": str(
                                param_set.model_dump()
                            ),  # Using model_dump instead of dict
                        },
                    )
                    feature_views.append(feature_view)

                except Exception as e:
                    logger.error(
                        f"Failed to create feature view for {feature_def.name}: {str(e)}"
                    )
                    continue

        return feature_views
