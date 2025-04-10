from typing import List
from feast import FeatureView, Field
from feast.types import Float32
from datetime import timedelta

from ai_trading.config.feature_config import FeaturesConfig
from ai_trading.preprocess.feature.feature_class_registry import FeatureClassRegistry
from feast.data_source import DataSource


class FeatureViewFactory:
    def __init__(
        self,
        config: FeaturesConfig,
        registry: FeatureClassRegistry,
        data_source: DataSource,
    ):
        self.config = config
        self.registry = registry
        self.data_source = data_source

    def create_feature_views(self) -> List[FeatureView]:
        feature_views = []

        for feature_def in self.config.feature_definitions:
            if not feature_def.enabled:
                continue

            for param_set in feature_def.parsed_parameter_sets:
                if not param_set.enabled:
                    continue

                feature_class = self.registry.feature_class_map[feature_def.name]
                field_names = feature_class.feature_names(param_set)

                fv = FeatureView(
                    name=f"{feature_def.name}_{param_set.hash_id()}",
                    entities=[],  # Add your entity here if needed
                    ttl=timedelta(days=365),
                    schema=[Field(name=col, dtype=Float32) for col in field_names],
                    source=self.data_source,
                )
                feature_views.append(fv)

        return feature_views
