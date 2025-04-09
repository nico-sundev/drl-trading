import dask
import dask.dataframe as dd
from ai_trading.preprocess.feature.feature_aggregator import FeatureAggregator

class DaskFeatureComputer:
    def __init__(self, config, class_registry):
        self.config = config
        self.class_registry = class_registry

    def compute(self, df: dd.DataFrame) -> dd.DataFrame:
        # Split work by feature + param_set
        aggregator = FeatureAggregator(df, self.config, self.class_registry)
        return dask.delayed(aggregator.compute)().compute()
