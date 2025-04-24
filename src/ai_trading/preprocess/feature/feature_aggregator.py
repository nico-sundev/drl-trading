import logging

from pandas import DataFrame, concat

from ai_trading.config.feature_config import FeaturesConfig
from ai_trading.model.asset_price_dataset import AssetPriceDataSet
from ai_trading.preprocess.feast.feast_service import FeastService
from ai_trading.preprocess.feature.feature_class_registry import FeatureClassRegistry

logger = logging.getLogger(__name__)


class FeatureAggregator:
    """
    Aggregates and computes features for asset price datasets.

    This class is responsible for:
    1. Computing features for asset price datasets
    2. Using the feature store to retrieve previously computed features
    3. Storing newly computed features in the feature store
    """

    def __init__(
        self,
        asset_data: AssetPriceDataSet,
        symbol: str,
        config: FeaturesConfig,
        class_registry: FeatureClassRegistry,
        feast_service: FeastService,
    ) -> None:
        """
        Initialize the FeatureAggregator with dataset and configuration.

        Args:
            asset_data: Dataset containing asset price information
            symbol: The trading symbol being processed
            config: Configuration for feature definitions
            class_registry: Registry of feature classes
            feast_service: Service for interacting with the Feast feature store
        """
        self.asset_data = asset_data
        self.config = config
        self.symbol = symbol
        self.class_registry = class_registry
        self.feast_service = feast_service

    def compute(self) -> DataFrame:
        """
        Compute features for the asset price dataset.

        For each enabled feature and parameter set:
        1. Try to retrieve from feature store if available
        2. Compute if not available
        3. Store computed features in feature store

        Returns:
            DataFrame: Combined DataFrame containing all computed features
        """
        feature_results = []

        for feature in self.config.feature_definitions:
            if not feature.enabled:
                continue

            for param_set in feature.parsed_parameter_sets:
                if not param_set.enabled:
                    continue

                # Create feature instance
                feature_class = self.class_registry.feature_class_map[feature.name]
                feature_instance = feature_class(
                    source=self.asset_data.asset_price_dataset, config=param_set
                )

                feature_name = feature_instance.get_feature_name()
                param_hash = param_set.hash_id()
                sub_feature_names = feature_instance.get_sub_features_names()

                # Try to get features from store first
                historical_features = self.feast_service.get_historical_features(
                    feature_name=feature_name,
                    param_hash=param_hash,
                    sub_feature_names=sub_feature_names,
                )

                if historical_features is not None:
                    feature_results.append(historical_features)
                    continue

                # Compute features if not found in store
                logger.info(
                    f"Computing features for {feature.name} with params {param_set}"
                )
                feature_df = feature_instance.compute()

                # Store computed features if feature store is enabled
                if self.feast_service.is_enabled():
                    self.feast_service.store_computed_features(
                        feature_df=feature_df,
                        feature_name=feature_name,
                        param_hash=param_hash,
                        sub_feature_names=sub_feature_names,
                    )

                feature_results.append(feature_df)

        if not feature_results:
            return DataFrame()

        # Combine all feature results
        return concat(
            [df.set_index("Time") for df in feature_results], axis=1
        ).reset_index()
