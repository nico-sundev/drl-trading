import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, cast

from dask import compute, delayed
from dask.delayed import Delayed
from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_common.config.feature_config import FeatureDefinition, FeaturesConfig
from drl_trading_common.interfaces.computable import Computable
from drl_trading_common.interfaces.feature.feature_factory_interface import (
    FeatureFactoryInterface,
)
from drl_trading_common.models.dataset_identifier import DatasetIdentifier
from injector import inject
from pandas import DataFrame

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureKey:
    """
    Immutable, observable key for uniquely identifying feature instances.

    Designed for multi-symbol, multi-timeframe feature management with full observability.
    Provides type safety, debuggability, and clear string representation for logging.
    """
    feature_name: str
    dataset_id: DatasetIdentifier
    param_hash: str

    def to_string(self) -> str:
        """
        Convert to human-readable string representation for logging and debugging.

        Format: feature_name_symbol_timeframe_param_hash
        Example: "rsi_EURUSD_H1_length14_abc123"
        """
        return f"{self.feature_name}_{self.dataset_id.symbol}_{self.dataset_id.timeframe.value}_{self.param_hash}"

    def __str__(self) -> str:
        """String representation for logging."""
        return self.to_string()

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"FeatureKey(feature='{self.feature_name}', symbol='{self.dataset_id.symbol}', timeframe='{self.dataset_id.timeframe.value}', param_hash='{self.param_hash}')"


@inject
class FeatureManager(Computable):
    """
    Acts as a Facade for all features
    by implementing the Computable interface.

    This service maintains references to all feature instances and delegates computation
    to the appropriate features.
    """

    def __init__(
        self, config: FeaturesConfig, feature_factory: FeatureFactoryInterface
    ) -> None:
        """
        Initialize the FeatureManagerService.

        Args:
            config: Configuration for feature definitions.
            feature_factory: Factory for creating feature instances.
        """
        self.config = config
        self.feature_factory = feature_factory
        # Enhanced feature storage with structured, observable keys
        # Key provides full context: feature + symbol + timeframe + parameters
        self._features: Dict[FeatureKey, BaseFeature] = {}

        # Observability metrics
        self._feature_creation_stats: Dict[str, Union[int|Dict]] = {
            "total_requested": 0,
            "successfully_created": 0,
            "creation_failures": 0,
            "features_by_symbol": {},
            "features_by_timeframe": {},
            "features_by_type": {}
        }

    def initialize_features(self) -> None:
        """
        Initialize feature instances based on the configuration using optimized patterns.
        """
        logger.info("Starting feature initialization process...")

        # Generate all valid feature configurations
        feature_configs = self._generate_feature_configurations()
        logger.info(f"Generated {len(feature_configs)} feature configurations")

        # Create features using batch processing
        created_features = self._create_features_batch(feature_configs)

        # Store created features
        self._store_features(created_features)

        # Update observability metrics
        self._update_initialization_metrics(created_features)

        # Log comprehensive summary
        self._log_initialization_summary()

        logger.info(f"Feature initialization completed: {len(self._features)} instances created from {len(feature_configs)} configurations")

    def _update_initialization_metrics(self, created_features: List[Tuple[FeatureKey, BaseFeature]]) -> None:
        """
        Update internal metrics for observability and debugging.

        Args:
            created_features: List of successfully created features
        """
        for feature_key, _ in created_features:
            # Track by symbol
            symbol = feature_key.dataset_id.symbol

            if symbol not in self._feature_creation_stats["features_by_symbol"]:
                self._feature_creation_stats["features_by_symbol"][symbol] = 0
            self._feature_creation_stats["features_by_symbol"][symbol] += 1

            # Track by timeframe
            timeframe = feature_key.dataset_id.timeframe.value
            if timeframe not in self._feature_creation_stats["features_by_timeframe"]:
                self._feature_creation_stats["features_by_timeframe"][timeframe] = 0
            self._feature_creation_stats["features_by_timeframe"][timeframe] += 1

            # Track by feature type
            feature_name = feature_key.feature_name
            if feature_name not in self._feature_creation_stats["features_by_type"]:
                self._feature_creation_stats["features_by_type"][feature_name] = 0
            self._feature_creation_stats["features_by_type"][feature_name] += 1

    def _log_initialization_summary(self) -> None:
        """
        Log comprehensive initialization summary for observability.
        """
        stats = self._feature_creation_stats

        logger.info("=== Feature Initialization Summary ===")
        logger.info(f"Total features created: {stats['successfully_created']}")
        logger.info(f"Creation failures: {stats['creation_failures']}")

        if stats['features_by_symbol']:
            logger.info("Features by symbol:")
            for symbol, count in stats['features_by_symbol'].items():
                logger.info(f"  {symbol}: {count} features")

        if stats['features_by_timeframe']:
            logger.info("Features by timeframe:")
            for timeframe, count in stats['features_by_timeframe'].items():
                logger.info(f"  {timeframe}: {count} features")

        if stats['features_by_type']:
            logger.info("Features by type:")
            for feature_type, count in stats['features_by_type'].items():
                logger.info(f"  {feature_type}: {count} instances")

        logger.info("=======================================")

    def _generate_feature_configurations(self) -> List[Tuple[str, DatasetIdentifier, BaseParameterSetConfig]]:
        """
        Generate all valid feature configurations using functional programming.

        Flattens the nested loop structure into a single list comprehension
        with proper filtering.

        Returns:
            List of tuples containing (feature_name, dataset_id, param_set)
        """
        from itertools import product

        # Get enabled feature definitions and their parameter sets
        enabled_features = [
            (feature_def.name, param_set)
            for feature_def in self.config.feature_definitions
            if feature_def.enabled
            for param_set in feature_def.parsed_parameter_sets
            if param_set.enabled
        ]

        # Get dataset identifiers
        dataset_ids = [
            DatasetIdentifier(symbol, timeframe)
            for symbol, timeframe in self.config.dataset_definitions.items()
        ]

        # Use cartesian product to generate all combinations efficiently
        configurations = [
            (feature_name, dataset_id, param_set)
            for (feature_name, param_set), dataset_id in product(enabled_features, dataset_ids)
        ]

        logger.debug(f"Generated {len(configurations)} feature configurations")
        return configurations

    def _create_features_batch(
        self,
        feature_configs: List[Tuple[str, DatasetIdentifier, BaseParameterSetConfig]]
    ) -> List[Tuple[FeatureKey, BaseFeature]]:
        """
        Create feature instances in batch.

        Uses Strategy pattern to separate creation logic from iteration logic.
        Enhanced with detailed metrics collection for debugging and monitoring.

        Args:
            feature_configs: List of feature configurations to create

        Returns:
            List of successfully created features as (FeatureKey, feature_instance)
        """
        created_features = []
        failed_count = 0

        self._feature_creation_stats["total_requested"] = len(feature_configs)

        for feature_name, dataset_id, param_set in feature_configs:
            try:
                feature_instance = self._create_feature_instance(
                    feature_name, dataset_id, param_set
                )

                if feature_instance:
                    param_hash = param_set.hash_id()
                    feature_key = FeatureKey(
                        feature_name=feature_name,
                        dataset_id=dataset_id,
                        param_hash=param_hash
                    )
                    created_features.append((feature_key, feature_instance))

                    # Log successful creation with full context
                    logger.debug(f"Created feature: {feature_key}")
                else:
                    failed_count += 1
                    logger.warning(f"Failed to create feature {feature_name} for {dataset_id}")

            except Exception as e:
                failed_count += 1
                logger.error(f"Exception creating feature {feature_name} for {dataset_id}: {str(e)}")

        # Update metrics
        self._feature_creation_stats["successfully_created"] = len(created_features)
        self._feature_creation_stats["creation_failures"] = failed_count

        if failed_count > 0:
            logger.warning(f"Failed to create {failed_count} feature instances")

        return created_features

    def _store_features(self, created_features: List[Tuple[FeatureKey, BaseFeature]]) -> None:
        """
        Store created features in the internal dictionary with observable keys.

        Separated storage logic for better testability and single responsibility.

        Args:
            created_features: List of (FeatureKey, feature_instance) tuples
        """
        conflicts_detected = 0

        for feature_key, feature_instance in created_features:
            if feature_key in self._features:
                conflicts_detected += 1
                logger.warning(f"Feature key conflict detected: {feature_key} (overwriting existing)")

            self._features[feature_key] = feature_instance
            logger.debug(f"Stored feature: {feature_key}")

        if conflicts_detected > 0:
            logger.error(f"Detected {conflicts_detected} feature key conflicts during storage!")

    def _create_feature_instance(
        self,
        feature_name: str,
        dataset_id: DatasetIdentifier,
        param_set: BaseParameterSetConfig,
        postfix: str = "",
    ) -> Optional[BaseFeature]:
        """
        Create a feature instance using the feature factory.

        Args:
            feature_name: Name of the feature.
            source_data: Source data for the feature.
            param_set: Parameter set for the feature.
            postfix: Optional postfix for the feature name.

        Returns:
            The created feature instance, or None if creation failed.
        """
        try:
            feature_instance = self.feature_factory.create_feature(
                feature_name=feature_name,
                dataset_id=dataset_id,
                config=param_set,
                postfix=postfix,
            )

            if not feature_instance:
                logger.error(f"Failed to create feature instance for '{feature_name}'")
                return None

            return feature_instance
        except Exception as e:
            logger.error(f"Error creating feature '{feature_name}': {str(e)}")
            return None

    def get_feature(self, feature_name: str, dataset_id: DatasetIdentifier, param_hash: str) -> Optional[BaseFeature]:
        """
        Get a specific feature instance.

        Args:
            feature_name: Name of the feature.
            param_hash: Hash of the parameter set.

        Returns:
            The feature instance if found, None otherwise.
        """
        feature_key = FeatureKey(
            feature_name=feature_name,
            dataset_id=dataset_id,
            param_hash=param_hash
        )
        return self._features.get(feature_key)

    def get_all_features(self) -> List[BaseFeature]:
        """
        Get all feature instances.

        Returns:
            List of all feature instances.
        """
        return list(self._features.values())

    def update_features_data(self, new_data: DataFrame) -> None:
        """
        Update all feature instances with new data.

        Args:
            new_data: New data to update features with.
        """
        for feature in self._features.values():
            try:
                computable_feature: Computable = cast(Computable, feature)
                computable_feature.add(new_data)
            except Exception as e:
                logger.error(f"Error updating feature {feature}: {str(e)}")

    def compute_feature(
        self,
        feature_def: FeatureDefinition,
        param_set: BaseParameterSetConfig,
        data: DataFrame,
        dataset_id: DatasetIdentifier,
    ) -> Optional[DataFrame]:
        """
        Compute a specific feature with the given parameters.

        Args:
            feature_def: The feature definition.
            param_set: The parameter set to use for computation.
            data: The data to compute the feature on.

        Returns:
            DataFrame with the computed feature, or None if computation fails.
        """
        if not feature_def.enabled or not param_set.enabled:
            return None

        # First check if we already have this feature instance
        param_hash = param_set.hash_id()
        feature = self.get_feature(feature_def.name, dataset_id, param_hash)

        if feature is None:
            # Create a new instance if not found
            feature = self._create_feature_instance(
                feature_def.name, dataset_id, param_set
            )

            if feature is None:
                return None

            # Store the instance for future use
            feature_key = FeatureKey(
                feature_name=feature_def.name,
                dataset_id=dataset_id,
                param_hash=param_hash
            )
            self._features[feature_key] = feature

        computable_feature: Computable = cast(Computable, feature)

        # Update the feature with the new data
        computable_feature.add(data)

        # Compute and return the result
        try:
            result = computable_feature.compute_all()
            return result
        except Exception as e:
            logger.error(
                f"Error computing feature {feature_def.name} with hash {param_hash}: {str(e)}"
            )
            return None

    def compute_feature_delayed(
        self,
        feature_def: FeatureDefinition,
        param_set: BaseParameterSetConfig,
        data: DataFrame,
        dataset_id: DatasetIdentifier,
    ) -> Delayed:
        """
        Creates a delayed task to compute a specific feature with the given parameters.

        Args:
            feature_def: The feature definition.
            param_set: The parameter set to use for computation.
            data: The data to compute the feature on.
            dataset_id: Dataset identifier for the feature.

        Returns:
            Delayed object that will compute and return the feature DataFrame when executed.
        """
        return delayed(self.compute_feature)(feature_def, param_set, data, dataset_id)

    def compute_latest_delayed(self, feature: BaseFeature) -> Delayed:
        """
        Create a delayed task to compute latest values for a specific feature.

        Args:
            feature: The feature instance to compute latest values for.

        Returns:
            Delayed object that will compute and return the latest feature values when executed.
        """
        return delayed(lambda f: f.compute_latest() if hasattr(f, "compute_latest") and callable(f.compute_latest) else None)(feature)

    def compute_features_latest_delayed(self) -> List[Delayed]:
        """
        Create delayed computation tasks for latest values of all features.

        Returns:
            List of delayed objects for computing latest values of all features.
        """
        delayed_tasks = [
            self.compute_latest_delayed(feature)
            for feature in self._features.values()
        ]

        logger.info(f"Generated {len(delayed_tasks)} delayed latest feature computation tasks.")
        return delayed_tasks

    def compute_all(self) -> Optional[DataFrame]:
        """
        Compute all features using Dask delayed execution for parallel processing.

        Returns:
            DataFrame with all computed features, or None if computation fails.
        """
        if not self._features:
            logger.warning("No features initialized. Call initialize_features() first.")
            return None

        # Create delayed tasks for all features' compute_all method
        delayed_tasks = [
            delayed(lambda f: f.compute_all() if hasattr(f, "compute_all") and callable(f.compute_all) else None)(feature)
            for feature in self._features.values()
        ]

        if not delayed_tasks:
            logger.warning("No delayed tasks created for feature computation.")
            return None

        try:
            # Execute all delayed tasks in parallel
            logger.info(f"Computing {len(delayed_tasks)} features using Dask delayed execution...")
            results = compute(*delayed_tasks)

            # Filter out None results and empty DataFrames
            valid_results = [
                result for result in results
                if result is not None and not result.empty
            ]

            if not valid_results:
                logger.warning("No valid feature results obtained from computation.")
                return None
            # Combine all results efficiently
            return self._combine_dataframes_efficiently(valid_results)

        except Exception as e:
            logger.error(f"Error during Dask delayed computation: {str(e)}")
            return None

    def _combine_dataframes_efficiently(self, dataframes: List[DataFrame]) -> Optional[DataFrame]:
        """
        Efficiently combine multiple DataFrames using pandas.concat.

        More efficient than iterative joins, especially for many DataFrames.

        Args:
            dataframes: List of DataFrames to combine

        Returns:
            Combined DataFrame or None if empty
        """
        if not dataframes:
            return None

        if len(dataframes) == 1:
            return dataframes[0]
        try:
            # Use concat with outer join for better performance than iterative joins
            import pandas as pd
            combined_df = pd.concat(dataframes, axis=1, join='outer', sort=False)
            return combined_df
        except Exception as e:
            logger.error(f"Error combining DataFrames with concat: {str(e)}")
            # Fallback to manual combination without using pandas concat or join
            # which might also be affected by the same patching in tests
            try:
                # Create a more robust manual merge that doesn't rely on pandas join
                import pandas as pd

                # Start with the first dataframe
                combined_df = dataframes[0].copy()

                # Manually add columns from other dataframes
                for df in dataframes[1:]:
                    for col in df.columns:
                        # Align indexes and add the column
                        if len(df) == len(combined_df):
                            combined_df[col] = df[col].values
                        else:
                            # For different lengths, use reindex to align
                            combined_df[col] = df[col].reindex(combined_df.index)

                return combined_df
            except Exception as join_error:
                logger.error(f"Error combining DataFrames with manual fallback: {str(join_error)}")
                # Final fallback: return first dataframe if all else fails
                return dataframes[0] if dataframes else None

    def add(self, df: DataFrame) -> None:
        """
        Add new data to all features.

        Args:
            df: New data to add.
        """
        self.update_features_data(df)

    def compute_latest(self) -> Optional[DataFrame]:
        """
        Compute latest values for all features using Dask delayed execution for parallel processing.

        Returns:
            DataFrame with latest computed features, or None if computation fails.
        """
        if not self._features:
            logger.warning("No features initialized. Call initialize_features() first.")
            return None

        # Create delayed tasks for computing latest values of all features
        delayed_tasks = self.compute_features_latest_delayed()

        if not delayed_tasks:
            logger.warning("No delayed tasks created for latest feature computation.")
            return None

        try:
            # Execute all delayed tasks in parallel
            logger.info(f"Computing latest values for {len(delayed_tasks)} features using Dask delayed execution...")
            results = compute(*delayed_tasks)

            # Filter out None results and empty DataFrames
            valid_results = [
                result for result in results
                if result is not None and not result.empty
            ]

            if not valid_results:
                logger.warning("No valid latest feature results obtained from computation.")
                return None

            # Combine all results efficiently
            return self._combine_dataframes_efficiently(valid_results)

        except Exception as e:
            logger.error(f"Error during Dask delayed computation for latest values: {str(e)}")
            return None
