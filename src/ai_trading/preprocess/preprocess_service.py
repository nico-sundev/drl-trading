from typing import List, Optional

import dask
from dask import delayed
from pandas import DataFrame

from ai_trading.config.feature_config import FeaturesConfig, FeatureStoreConfig
from ai_trading.data_set_utils.merge_service import MergeService
from ai_trading.data_set_utils.util import separate_computed_datasets
from ai_trading.model.asset_price_dataset import AssetPriceDataSet
from ai_trading.model.computed_dataset_container import ComputedDataSetContainer
from ai_trading.preprocess.feature.feature_aggregator import FeatureAggregator
from ai_trading.preprocess.feature.feature_class_registry import FeatureClassRegistry


class PreprocessService:
    def __init__(
        self,
        datasets: List[AssetPriceDataSet],
        features_config: FeaturesConfig,
        feature_class_registry: FeatureClassRegistry,
        feature_store_config: Optional[FeatureStoreConfig] = None,
    ) -> None:
        self.features_config = features_config
        self.feature_class_registry = feature_class_registry
        self.datasets = datasets
        self.feature_store_config = feature_store_config

    def preprocess_data(self) -> DataFrame:
        # Parallelize feature computation for each dataset using delayed
        feature_results = [
            delayed(self.compute_feature)(dataset) for dataset in self.datasets
        ]

        # Compute all delayed tasks in parallel
        asset_price_datasets: List[ComputedDataSetContainer] = dask.compute(
            *feature_results
        )[0]
        base_dataset, other_datasets = separate_computed_datasets(asset_price_datasets)

        # Initialize a merged dataset with the base_dataset first
        merged_result = base_dataset.computed_dataframe

        # Iterate through other_datasets and merge with base_dataset one by one
        for dataset in other_datasets:
            # Create an instance of the MergingService for each pair of datasets
            merger = MergeService(merged_result, dataset.computed_dataframe)
            # Perform the merge and update the merged_result
            merged_result = merger.merge_timeframes()
        return merged_result

    def compute_feature(self, dataset: AssetPriceDataSet) -> ComputedDataSetContainer:
        """
        Computes the features for a given dataset.

        Args:
            dataset: The dataset to compute features for

        Returns:
            A container with the original dataset and computed features
        """
        feature_aggregator = FeatureAggregator(
            dataset.asset_price_dataset,
            self.features_config,
            self.feature_class_registry,
            self.feature_store_config,
        )
        computed_data = feature_aggregator.compute()
        return ComputedDataSetContainer(dataset, computed_data)
