from pandas import DataFrame, concat
from ai_trading.config.feature_config import FeaturesConfig
from ai_trading.preprocess.feature.collection.feature_mapper import FEATURE_MAP


class FeatureAggregator:

    def __init__(self, source_df: DataFrame, config: FeaturesConfig):
        self.source_df = source_df
        self.config = config

    def compute(self) -> DataFrame:
        feature_results = []
        for feature in self.config.feature_definitions:
            if feature.name == "rsi":
                for param_set in feature.parsed_parameter_sets:
                    feature_class = FEATURE_MAP[feature.name]
                    feature_instance = feature_class(self.source_df)
                    feature_df = feature_instance.compute(param_set)
                    feature_results.append(feature_df)

        return concat(
            [df.set_index("Time") for df in feature_results], axis=1
        ).reset_index()

    # def compute_derivatives(underlyingFeature: DataFrame) -> list[DataFrame]:
