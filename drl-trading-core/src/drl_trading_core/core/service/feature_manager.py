import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, cast

from dask import compute, delayed
from dask.delayed import Delayed
from injector import inject
from pandas import DataFrame

from drl_trading_core.core.port.base_feature import BaseFeature
from drl_trading_core.core.model.feature.feature_metadata import FeatureMetadata
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_core.core.port.computable import Computable
from drl_trading_core.core.service.feature.feature_factory_interface import (
    IFeatureFactory,
)
from drl_trading_common.core.model.dataset_identifier import DatasetIdentifier
from drl_trading_core.core.config.feature_computation_config import (
    FeatureComputationConfig,
)
from drl_trading_common.core.model.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_core.core.model.feature_computation_request import (
    FeatureComputationRequest,
)
from drl_trading_core.core.model.feature_definition import FeatureDefinition
from drl_trading_core.core.service.feature_parameter_set_parser import FeatureParameterSetParser

logger = logging.getLogger(__name__)

NO_CONFIG_HASH = "no_config"


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
        self,
        feature_factory: IFeatureFactory,
        feature_parameter_set_parser: FeatureParameterSetParser,
        feature_computation_config: FeatureComputationConfig,
    ) -> None:
        """
        Initialize the FeatureManagerService.

        Args:
            feature_factory: Factory for creating feature instances.
            feature_parameter_set_parser: Parser for feature definitions.
            feature_computation_config: Configuration for feature computation parallelization.
        """
        self.feature_factory = feature_factory
        self.feature_parameter_set_parser = feature_parameter_set_parser
        self.feature_computation_config = feature_computation_config

        # Enhanced feature storage with structured, observable keys
        # Key provides full context: feature + symbol + timeframe + parameters
        self._features: Dict[FeatureKey, BaseFeature] = {}
        self._features_name_role_cache: Dict[str, FeatureRoleEnum] = {}

        # Observability metrics
        self._feature_creation_stats: Dict[str, Union[int, Dict[str, int]]] = {
            "total_requested": 0,
            "successfully_created": 0,
            "creation_failures": 0,
            "features_by_symbol": {},
            "features_by_timeframe": {},
            "features_by_type": {},
        }

    def _update_features(self, request: FeatureComputationRequest) -> None:
        """
        Initialize feature instances based on the configuration using optimized patterns.
        """
        logger.info("Starting feature initialization process...")

        # Populate parsed feature configurations
        self.feature_parameter_set_parser.parse_feature_definitions(
            request.feature_definitions
        )

        # Generate all valid feature configurations
        feature_configs = self._generate_feature_configurations(request)
        logger.debug(f"Found {len(feature_configs)} feature configuration combinations")

        # Create features using batch processing
        created_features = self._create_features_batch(feature_configs)

        # Store created features
        self._store_features(created_features)

        # Update observability metrics
        self._update_initialization_metrics(created_features)

        # Log comprehensive summary
        self._log_initialization_summary()

        logger.info(
            f"Feature initialization completed: {len(self._features)} instances created from {len(feature_configs)} configurations"
        )

    # def get_features_by_role(self, role: FeatureRoleEnum):
    #     """
    #     Get all features by their role.
    #     Args:
    #         role: The role to filter features by.
    #     Returns:
    #         List[BaseFeature]: List of features matching the specified role.
    #     """
    #     return [
    #         feature
    #         for feature in self._features.values()
    #         if feature.get_feature_role() == role
    #     ]

    def _update_initialization_metrics(
        self, created_features: List[Tuple[FeatureKey, BaseFeature]]
    ) -> None:
        """
        Update internal metrics for observability and debugging.

        Args:
            created_features: List of successfully created features
        """
        for feature_key, _ in created_features:
            # Track by symbol
            symbol = feature_key.dataset_id.symbol

            features_by_symbol = cast(
                Dict[str, int], self._feature_creation_stats["features_by_symbol"]
            )
            if symbol not in features_by_symbol:
                features_by_symbol[symbol] = 0
            features_by_symbol[symbol] += 1

            # Track by timeframe
            timeframe = feature_key.dataset_id.timeframe.value
            features_by_timeframe = cast(
                Dict[str, int], self._feature_creation_stats["features_by_timeframe"]
            )
            if timeframe not in features_by_timeframe:
                features_by_timeframe[timeframe] = 0
            features_by_timeframe[timeframe] += 1

            # Track by feature type
            feature_name = feature_key.feature_name
            features_by_type = cast(
                Dict[str, int], self._feature_creation_stats["features_by_type"]
            )
            if feature_name not in features_by_type:
                features_by_type[feature_name] = 0
            features_by_type[feature_name] += 1

    def _log_initialization_summary(self) -> None:
        """
        Log comprehensive initialization summary for observability.
        """
        stats = self._feature_creation_stats

        logger.debug("=== Feature Initialization Summary ===")
        logger.debug(f"Total features created: {stats['successfully_created']}")
        logger.debug(f"Creation failures: {stats['creation_failures']}")

        features_by_symbol = cast(Dict[str, int], stats["features_by_symbol"])
        if features_by_symbol:
            logger.debug("Features by symbol:")
            for symbol, count in features_by_symbol.items():
                logger.debug(f"  {symbol}: {count} features")

        features_by_timeframe = cast(Dict[str, int], stats["features_by_timeframe"])
        if features_by_timeframe:
            logger.debug("Features by timeframe:")
            for timeframe, count in features_by_timeframe.items():
                logger.debug(f"  {timeframe}: {count} features")

        features_by_type = cast(Dict[str, int], stats["features_by_type"])
        if features_by_type:
            logger.debug("Features by type:")
            for feature_type, count in features_by_type.items():
                logger.debug(f"  {feature_type}: {count} instances")

        logger.debug("=======================================")

    def _generate_feature_configurations(
        self, request: FeatureComputationRequest
    ) -> List[Tuple[str, DatasetIdentifier, Optional[BaseParameterSetConfig]]]:
        """
        Generate all combinations of features that need to be computed.

        Takes the feature definitions from the configuration and creates individual
        computation tasks for each enabled feature on each symbol/timeframe combination.
        Features with multiple parameter sets (like RSI with different lengths) will
        generate multiple configurations, while simple features (like close price)
        generate one configuration per dataset.

        Returns:
            List of tuples where each tuple represents one feature computation task:
            - feature_name: The name of the feature to compute (e.g., "rsi", "close_price")
            - dataset_id: Which symbol and timeframe to compute it for
            - config: The parameters for this feature instance, or None for simple features

        Example:
            If configured with RSI (lengths 14,21) and close_price on BTCUSD/H1 and ETHUSD/M5:
            Returns 6 configurations:
            - RSI with length=14 on BTCUSD/H1
            - RSI with length=21 on BTCUSD/H1
            - RSI with length=14 on ETHUSD/M5
            - RSI with length=21 on ETHUSD/M5
            - close_price on BTCUSD/H1 (no config needed)
            - close_price on ETHUSD/M5 (no config needed)
        """
        from itertools import product

        # Get enabled feature definitions and their parameter sets
        enabled_features: List[Tuple[str, Optional[BaseParameterSetConfig]]] = []
        for feature_def in request.feature_definitions:
            if not feature_def.enabled:
                continue

            # If feature has no parameter sets, include it with None config
            if not feature_def.parsed_parameter_sets:
                enabled_features.append((feature_def.name, None))
            else:
                # Include enabled parameter sets
                for param_set in feature_def.parsed_parameter_sets.values():
                    if param_set.enabled:
                        enabled_features.append((feature_def.name, param_set))

        # Use cartesian product to generate all combinations efficiently
        configurations = [
            (feature_name, dataset_id, param_set)
            for (feature_name, param_set), dataset_id in product(
                enabled_features, [request.dataset_id]
            )
        ]

        logger.debug(f"Generated {len(configurations)} feature configurations")
        return configurations

    def _create_features_batch(
        self,
        feature_configs: List[
            Tuple[str, DatasetIdentifier, Optional[BaseParameterSetConfig]]
        ],
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
                # Generate hash for the parameter set (handle None case)
                param_hash = param_set.hash_id() if param_set else NO_CONFIG_HASH
                feature_key = FeatureKey(
                    feature_name=feature_name,
                    dataset_id=dataset_id,
                    param_hash=param_hash,
                )
                if self._features.get(feature_key):
                    continue

                feature_instance = self._create_feature_instance(
                    feature_name, dataset_id, param_set
                )

                if feature_instance:
                    created_features.append((feature_key, feature_instance))
                    # Log successful creation with full context
                    logger.debug(f"Created feature: {feature_key}")
                else:
                    failed_count += 1
                    logger.warning(
                        f"Failed to create feature {feature_name} for {dataset_id}"
                    )

            except Exception as e:
                failed_count += 1
                logger.error(
                    f"Exception creating feature {feature_name} for {dataset_id}: {str(e)}"
                )

        # Update metrics
        self._feature_creation_stats["successfully_created"] = len(created_features)
        self._feature_creation_stats["creation_failures"] = failed_count

        if failed_count > 0:
            logger.warning(f"Failed to create {failed_count} feature instances")

        return created_features

    def _store_features(
        self, created_features: List[Tuple[FeatureKey, BaseFeature]]
    ) -> None:
        """
        Store created features in the internal dictionary with observable keys.

        Reuses existing features if they already exist to preserve warmup state.
        This allows features to be warmed up once and then reused for computation
        without recreating them and losing their state.

        Args:
            created_features: List of (FeatureKey, feature_instance) tuples
        """
        reused_count = 0
        stored_count = 0

        for feature_key, feature_instance in created_features:
            if feature_key in self._features:
                reused_count += 1
                logger.debug(f"Feature already exists, reusing: {feature_key}")
                # Skip storing - reuse existing feature to preserve warmup state
                continue

            self._features[feature_key] = feature_instance
            stored_count += 1
            logger.debug(f"Stored feature: {feature_key}")
            self._features_name_role_cache[feature_key.feature_name] = (
                feature_instance.get_metadata().feature_role
            )

        if reused_count > 0:
            logger.info(
                f"Reused {reused_count} existing features (preserving warmup state)"
            )

    def _create_feature_instance(
        self,
        feature_name: str,
        dataset_id: DatasetIdentifier,
        param_set: Optional[BaseParameterSetConfig],
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

    def _get_feature(
        self, feature_name: str, dataset_id: DatasetIdentifier, param_hash: str
    ) -> Optional[BaseFeature]:
        """
        Get a specific feature instance.

        Args:
            feature_name: Name of the feature.
            param_hash: Hash of the parameter set.

        Returns:
            The feature instance if found, None otherwise.
        """
        feature_key = FeatureKey(
            feature_name=feature_name, dataset_id=dataset_id, param_hash=param_hash
        )
        return self._features.get(feature_key)

    def _update_features_data(self, df: DataFrame) -> None:
        """
        Update all feature instances with new data.

        Args:
            df: New data to update features with.
        """
        for feature in self._features.values():
            try:
                computable_feature: Computable = cast(Computable, feature)
                computable_feature.update(df)
            except Exception as e:
                logger.error(f"Error updating feature {feature}: {str(e)}")

    def _compute_latest_delayed(self, feature: BaseFeature) -> Delayed:
        """
        Create a delayed task to compute latest values for a specific feature.

        Args:
            feature: The feature instance to compute latest values for.

        Returns:
            Delayed object that will compute and return the latest feature values when executed.
        """
        return cast(
            Delayed,
            delayed(
                lambda f: (
                    f.compute_latest()
                    if hasattr(f, "compute_latest") and callable(f.compute_latest)
                    else None
                )
            )(feature),
        )

    def _compute_features_latest_delayed(self) -> List[Delayed]:
        """
        Create delayed computation tasks for latest values of all features.

        Returns:
            List of delayed objects for computing latest values of all features.
        """
        delayed_tasks = [
            self._compute_latest_delayed(feature) for feature in self._features.values()
        ]

        logger.info(
            f"Generated {len(delayed_tasks)} delayed latest feature computation tasks."
        )
        return delayed_tasks

    def compute_all(self) -> Optional[DataFrame]:
        """
        Compute all features using Dask delayed execution for parallel processing.

        Uses the injected FeatureComputationConfig to configure Dask scheduler,
        workers, and memory limits for optimal CPU-bound feature computation.

        Returns:
            DataFrame with all computed features, or None if computation fails.
        """
        if not self._features:
            logger.warning("No features initialized. Call initialize_features() first.")
            return None

        # Create delayed tasks for all features' compute_all method
        delayed_tasks = [
            delayed(
                lambda f: (
                    f.compute_all()
                    if hasattr(f, "compute_all") and callable(f.compute_all)
                    else None
                )
            )(feature)
            for feature in self._features.values()
        ]

        if not delayed_tasks:
            logger.warning("No delayed tasks created for feature computation.")
            return None

        try:
            # Get Dask configuration from injected config
            dask_config = self.feature_computation_config.dask

            # Execute all delayed tasks with configured scheduler
            logger.info(
                f"Computing {len(delayed_tasks)} features using Dask "
                f"(scheduler={dask_config.scheduler}, workers={dask_config.num_workers})..."
            )

            # Pass scheduler configuration to compute()
            # For processes/threads schedulers, Dask handles worker pool management
            results = compute(
                *delayed_tasks,
                scheduler=dask_config.scheduler,
                num_workers=dask_config.num_workers,
            )

            # Filter out None results and empty DataFrames
            valid_results = [
                result for result in results if result is not None and not result.empty
            ]

            if not valid_results:
                logger.warning("No valid feature results obtained from computation.")
                return None
            # Combine all results efficiently
            return self._combine_dataframes_efficiently(valid_results)

        except Exception as e:
            logger.error(f"Error during Dask delayed computation: {str(e)}")
            return None

    def _combine_dataframes_efficiently(
        self, dataframes: List[DataFrame]
    ) -> Optional[DataFrame]:
        """
        Efficiently combine multiple DataFrames using pandas.concat.

        More efficient than iterative joins, especially for many DataFrames.
        Handles duplicate columns (like event_timestamp) by keeping only the first occurrence.

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

            combined_df = pd.concat(dataframes, axis=1, join="outer", sort=False)

            # Handle duplicate columns: keep only the first occurrence
            # This commonly happens with event_timestamp when multiple features include it
            if combined_df.columns.duplicated().any():
                duplicate_cols = combined_df.columns[
                    combined_df.columns.duplicated()
                ].unique()
                logger.debug(
                    f"Found duplicate columns after concat: {list(duplicate_cols)}"
                )
                # Keep first occurrence, drop subsequent duplicates
                combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
                logger.debug(
                    f"Removed duplicate columns, final columns: {list(combined_df.columns)}"
                )

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
                        # Skip columns that already exist (e.g., event_timestamp, symbol)
                        if col in combined_df.columns:
                            logger.debug(
                                f"Skipping duplicate column '{col}' during manual combination"
                            )
                            continue

                        # Align indexes and add the column
                        if len(df) == len(combined_df):
                            combined_df[col] = df[col].values
                        else:
                            # For different lengths, use reindex to align
                            combined_df[col] = df[col].reindex(combined_df.index)

                return combined_df
            except Exception as join_error:
                logger.error(
                    f"Error combining DataFrames with manual fallback: {str(join_error)}"
                )
                # Final fallback: return first dataframe if all else fails
                return dataframes[0] if dataframes else None

    # This has to be refactored soon
    def initialize_features(
        self,
        dataset_id: DatasetIdentifier,
        feature_definitions: List[FeatureDefinition],
    ) -> None:
        """
        Public method to initialize features based on the configuration.

        Args:
            request: Feature computation request containing definitions and dataset info.
        """

        self._update_features(
            FeatureComputationRequest(
                dataset_id=dataset_id,
                feature_definitions=feature_definitions,
                market_data=DataFrame(),  # Empty DataFrame for initialization
            )
        )

    def request_features_update(self, request: FeatureComputationRequest) -> None:
        """
        Public method to update features with new data.

        Args:
            new_data: New data to update features with.
            features_config: Configuration for the features.
        """

        self._update_features(request)
        self._update_features_data(request.market_data)

    def update(self, df: DataFrame) -> None:
        """
        Add new data to all features.

        Args:
            df: New data to add.
        """
        self._update_features_data(df)

    def compute_latest(self) -> Optional[DataFrame]:
        """
        Compute latest values for all features using Dask delayed execution for parallel processing.

        Uses the injected FeatureComputationConfig to configure Dask scheduler,
        workers, and memory limits for optimal CPU-bound feature computation.

        Returns:
            DataFrame with latest computed features, or None if computation fails.
        """
        if not self._features:
            logger.warning("No features initialized. Call initialize_features() first.")
            return None

        # Create delayed tasks for computing latest values of all features
        delayed_tasks = self._compute_features_latest_delayed()

        if not delayed_tasks:
            logger.warning("No delayed tasks created for latest feature computation.")
            return None

        try:
            # Get Dask configuration from injected config
            dask_config = self.feature_computation_config.dask

            # Execute all delayed tasks with configured scheduler
            logger.info(
                f"Computing latest values for {len(delayed_tasks)} features using Dask "
                f"(scheduler={dask_config.scheduler}, workers={dask_config.num_workers})..."
            )

            # Pass scheduler configuration to compute()
            results = compute(
                *delayed_tasks,
                scheduler=dask_config.scheduler,
                num_workers=dask_config.num_workers,
            )

            # Filter out None results and empty DataFrames
            valid_results = [
                result for result in results if result is not None and not result.empty
            ]

            if not valid_results:
                logger.warning(
                    "No valid latest feature results obtained from computation."
                )
                return None

            # Combine all results efficiently
            return self._combine_dataframes_efficiently(valid_results)

        except Exception as e:
            logger.error(
                f"Error during Dask delayed computation for latest values: {str(e)}"
            )
            return None

    def are_features_caught_up(self, reference_time: datetime) -> bool:
        """
        Check if ALL features are caught up based on the last available record time.

        Loops over all features and returns True only if ALL features are caught up.
        A feature is considered caught up if the time difference between its last
        record and the reference time is less than its configured timeframe duration.

        Args:
            reference_time: The current or target datetime to compare against

        Returns:
            True if ALL features are caught up, False if any feature is not caught up
            or if no features exist
        """
        if not self._features:
            logger.warning("No features initialized. Cannot determine catch-up status.")
            return False

        # Check each feature's catch-up status
        caught_up_count = 0
        total_features = len(self._features)

        for feature_key, feature in self._features.items():
            try:
                is_feature_caught_up = feature.are_features_caught_up(reference_time)
                if is_feature_caught_up:
                    caught_up_count += 1
                else:
                    logger.debug(f"Feature {feature_key} is not caught up")
            except Exception as e:
                logger.error(
                    f"Error checking catch-up status for feature {feature_key}: {str(e)}"
                )
                # If we can't determine status, consider it not caught up

        all_caught_up = caught_up_count == total_features

        if all_caught_up:
            logger.info(f"All {total_features} features are caught up")
        else:
            logger.warning(
                f"Only {caught_up_count}/{total_features} features are caught up"
            )

        return all_caught_up

    def is_feature_supported(self, feature_name: str) -> bool:
        """
        Check if a feature is supported by the feature factory.

        Args:
            feature_name: Name of the feature to validate

        Returns:
            True if feature is supported and can be created, False otherwise
        """
        try:
            return bool(self.feature_factory.is_feature_supported(feature_name))
        except Exception as e:
            logger.warning(f"Failed to validate feature {feature_name}: {e}")
            return False

    def validate_feature_definitions(
        self, feature_definitions: List[FeatureDefinition]
    ) -> Dict[str, bool]:
        """
        Validate multiple feature definitions for support.

        Args:
            feature_definitions: List of feature definitions to validate

        Returns:
            Dictionary mapping feature names to validation status
        """
        validation_results = {}

        for feature_def in feature_definitions:
            validation_results[feature_def.name] = self.is_feature_supported(
                feature_def.name
            )

        logger.debug(
            f"Validated {len(feature_definitions)} features: "
            f"{sum(validation_results.values())}/{len(validation_results)} supported"
        )

        return validation_results

    def get_feature_role(self, feature_name: str) -> Optional[FeatureRoleEnum]:
        """
        Get the role of a specific feature.

        Args:
            feature_name: Name of the feature

        Returns:
            The role of the feature, or None if not found
        """
        # First check the cache
        cached_feature_role = self._features_name_role_cache.get(feature_name)
        if cached_feature_role:
            return cached_feature_role
        logger.warning(f"Feature {feature_name} not found when retrieving role")
        return None

    def get_feature_metadata_list(
        self,
        feature_definitions: List[FeatureDefinition],
        dataset_id: DatasetIdentifier,
    ) -> Dict[FeatureRoleEnum, List[FeatureMetadata]]:
        """
        Get feature metadata for a list of feature definitions by searching the features cache.

        This method searches the existing feature cache for instances matching the dataset_id
        and feature definitions, extracting metadata from cached instances and grouping by feature role.

        Args:
            feature_definitions: List of feature definitions to get metadata for
            dataset_id: Dataset identifier to search for cached features

        Returns:
            Dict mapping FeatureRoleEnum to List of FeatureMetadata objects for features found in cache
        """
        logger.debug(
            f"Extracting metadata for {len(feature_definitions)} feature definitions from cache for {dataset_id}"
        )

        metadata_by_role: Dict[FeatureRoleEnum, List[FeatureMetadata]] = {}

        for feature_def in feature_definitions:
            try:
                # Handle features with parameter sets
                if feature_def.parsed_parameter_sets:
                    for param_set in feature_def.parsed_parameter_sets:
                        key = FeatureKey(
                            feature_def.name, dataset_id, param_set
                        )
                        feature = self._features.get(key)
                        if feature:
                            role = feature.get_metadata().feature_role
                            if role not in metadata_by_role:
                                metadata_by_role[role] = []
                            metadata_by_role[role].append(feature.get_metadata())
                            logger.debug(
                                f"Found cached metadata for feature '{feature_def.name}' with params in {dataset_id}, role {role}"
                            )
                else:
                    # Handle features without parameter sets
                    key = FeatureKey(feature_def.name, dataset_id, NO_CONFIG_HASH)
                    feature = self._features.get(key)
                    if feature:
                        role = feature.get_metadata().feature_role
                        if role not in metadata_by_role:
                            metadata_by_role[role] = []
                        metadata_by_role[role].append(feature.get_metadata())
                        logger.debug(
                            f"Found cached metadata for feature '{feature_def.name}' in {dataset_id}, role {role}"
                        )
                    else:
                        logger.warning(
                            f"No cached feature instance found for '{feature_def.name}' in {dataset_id}"
                        )

            except Exception as e:
                logger.error(
                    f"Error extracting metadata for feature '{feature_def.name}': {e}"
                )

        logger.debug(
            f"Successfully extracted metadata for {sum(len(v) for v in metadata_by_role.values())} features from cache, grouped by role"
        )
        return metadata_by_role
