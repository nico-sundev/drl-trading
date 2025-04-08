from pandas import DataFrame, concat
from ai_trading.config.feature_config import FeaturesConfig
from ai_trading.preprocess.feature.feature_class_registry import FeatureClassRegistry


class FeatureAggregator:

    def __init__(
        self,
        source_df: DataFrame,
        config: FeaturesConfig,
        class_registry: FeatureClassRegistry,
    ):
        self.source_df = source_df
        self.config = config
        self.class_registry = class_registry

    def compute(self) -> DataFrame:
        feature_results = []
        for feature in self.config.feature_definitions:
            if feature.enabled:
                for param_set in feature.parsed_parameter_sets:
                    feature_class = self.class_registry.feature_class_map[feature.name]
                    feature_instance = feature_class(self.source_df)
                    feature_df = feature_instance.compute(param_set)
                    feature_results.append(feature_df)

        return concat(
            [df.set_index("Time") for df in feature_results], axis=1
        ).reset_index()

    # def compute_derivatives(underlyingFeature: DataFrame) -> list[DataFrame]:
