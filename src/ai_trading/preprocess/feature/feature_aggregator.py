from pandas import DataFrame, concat
from ai_trading.config.feature_config import FeaturesConfig
from ai_trading.preprocess.feature.feature_factory import FeatureFactory


class FeatureAggregator:
    
    def __init__(self, feature_factory: FeatureFactory, config: FeaturesConfig):
        self.feature_factory = feature_factory
        self.config = config
        
    def compute(self) -> DataFrame:
        feature_dfs = [
            self.feature_factory.compute_macd_signals(
                self.config.macd.fast, self.config.macd.slow, self.config.macd.signal
            ),
            self.feature_factory.compute_ranges(self.config.range.lookback, self.config.range.wick_handle_strategy),
            *(self.feature_factory.compute_roc(length) for length in self.config.roc_lengths),
            *(self.feature_factory.compute_rsi(length) for length in self.config.rsi_lengths),
        ]

        # Ensure there are no empty DataFrames
        feature_dfs = [df for df in feature_dfs if not df.empty]

        if not feature_dfs:
            return DataFrame()  # Return empty DataFrame if there's nothing to merge

        # Concatenate all DataFrames along columns (axis=1), making sure they are aligned by "Time"
        # The assumption here is that all DataFrames have the same "Time" column
        merged_df = concat(
            feature_dfs, axis=1, join="inner"
        )  # Use 'inner' join to align based on "Time"

        return merged_df
    
    #def compute_derivatives(underlyingFeature: DataFrame) -> list[DataFrame]:
        