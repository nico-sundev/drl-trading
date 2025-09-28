import logging
from datetime import datetime

from injector import inject
from pandas import DataFrame, Series

from drl_trading_common.config.feature_config import FeaturesConfig
from drl_trading_core.core.service.feature_manager import FeatureManager
from drl_trading_preprocess.core.model.computation.feature_catchup_status import FeatureCatchupStatus

logger = logging.getLogger(__name__)


@inject
class FeatureComputingService:
    """Service for computing features on market data."""

    def __init__(self, feature_manager: FeatureManager) -> None:
        """Initialize the ComputingService with feature manager.

        Args:
            feature_manager: Service that manages feature instances.
        """
        self.feature_manager = feature_manager

    def compute_batch(self, data: DataFrame, features_config: FeaturesConfig) -> DataFrame:
        """Compute results on a batch of data.

        Args:
            data: Data to compute features for.
            features_config: Configuration for the features to compute.

        Returns:
            DataFrame with computed features.
        """
        # Update features with new data
        self.feature_manager.request_features_update(data, features_config)

        # Compute all features
        result = self.feature_manager.compute_all()
        if result is None:
            logger.warning("No results computed. Returning empty DataFrame.")
            return DataFrame()

        return result

    def compute_incremental(self, data_point: Series, features_config: FeaturesConfig) -> Series:
        """Compute results incrementally for a single data point.

        Args:
            data_point: A single data point to compute features for.
            features_config: Configuration for the features to compute.

        Returns:
            Series with computed features for the data point.
        """
        # Convert Series to DataFrame with a single row
        data_df = DataFrame([data_point])

        # Update features with new data point
        self.feature_manager.request_features_update(data_df, features_config)

        # Compute latest features
        result_df = self.feature_manager.compute_latest()

        if result_df is None or result_df.empty:
            logger.warning("No incremental results computed. Returning empty Series.")
            return Series()

        # Convert the result DataFrame's last row to a Series
        return result_df.iloc[-1] if not result_df.empty else Series()

    def check_catchup_status(self, reference_time: datetime) -> FeatureCatchupStatus:
        """Check detailed catch-up status for all features.

        Args:
            reference_time: The current or target datetime to compare against

        Returns:
            FeatureCatchupStatus with detailed information about each feature
        """
        if not hasattr(self.feature_manager, '_features') or not self.feature_manager._features:
            logger.warning("No features initialized. Cannot determine catch-up status.")
            return FeatureCatchupStatus(
                all_caught_up=False,
                caught_up_features=[],
                not_caught_up_features=[],
                total_features=0,
                catch_up_percentage=0.0,
                reference_time=reference_time
            )

        caught_up_features = []
        not_caught_up_features = []

        # Check each feature's catch-up status
        for feature_key, feature in self.feature_manager._features.items():
            try:
                is_feature_caught_up = feature.are_features_caught_up(reference_time)
                feature_name = str(feature_key)

                if is_feature_caught_up:
                    caught_up_features.append(feature_name)
                else:
                    not_caught_up_features.append(feature_name)
                    logger.debug(f"Feature {feature_name} is not caught up")

            except Exception as e:
                logger.error(f"Error checking catch-up status for feature {feature_key}: {str(e)}")
                # If we can't determine status, consider it not caught up
                not_caught_up_features.append(str(feature_key))

        total_features = len(caught_up_features) + len(not_caught_up_features)
        all_caught_up = len(not_caught_up_features) == 0
        catch_up_percentage = (len(caught_up_features) / total_features * 100) if total_features > 0 else 0.0

        status = FeatureCatchupStatus(
            all_caught_up=all_caught_up,
            caught_up_features=caught_up_features,
            not_caught_up_features=not_caught_up_features,
            total_features=total_features,
            catch_up_percentage=catch_up_percentage,
            reference_time=reference_time
        )

        if all_caught_up:
            logger.info(f"All {total_features} features are caught up")
        else:
            logger.warning(f"Only {len(caught_up_features)}/{total_features} features are caught up")
            logger.info(f"Features needing warmup: {not_caught_up_features}")

        return status
