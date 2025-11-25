import logging

import pandas as pd

from drl_trading_adapter.adapter.feature_store.provider import FeastProvider
from injector import inject
from pandas import DataFrame

from drl_trading_adapter.adapter.feature_store.util.feature_store_utilities import (
    get_feature_service_name,
)
from drl_trading_common.enum import FeatureRoleEnum
from drl_trading_core.core.dto.feature_service_metadata import (
    FeatureServiceMetadata,
)
from drl_trading_core.core.dto.offline_storage_request import OfflineStorageRequest
from drl_trading_core.core.mapper import FeatureViewNameMapper

from ...core.port import IFeatureStoreSavePort

logger = logging.getLogger(__name__)


@inject
class FeatureStoreSaveRepository(IFeatureStoreSavePort):
    """
    Repository for saving features to a feature store.

    This class focuses solely on Feast feature store operations,
    delegating the actual file/S3 storage to OfflineFeatureRepoInterface implementations.
    This follows the Single Responsibility Principle by separating:
    - Feature store orchestration (this class)
    - Storage backend operations (OfflineFeatureRepoInterface implementations)
    """

    def __init__(
        self,
        feast_provider: FeastProvider,
        feature_view_name_mapper: FeatureViewNameMapper,
    ):
        self.feast_provider = feast_provider
        self.feature_store = feast_provider.get_feature_store()
        self.offline_repo = feast_provider.get_offline_repo()
        self.feature_view_name_mapper = feature_view_name_mapper

    def store_computed_features_offline(
        self,
        request: OfflineStorageRequest,
    ) -> None:
        """
        Store computed features using the configured offline repository and
        create Feast feature views and services.

        Args:
            request: Offline storage request containing all necessary parameters
        """
        features_df = request.features_df
        feature_service_metadata = request.feature_service_metadata
        processing_context = request.processing_context
        requested_start_time = request.requested_start_time
        requested_end_time = request.requested_end_time
        symbol = request.symbol

        # Validate input
        if features_df.empty:
            logger.info(f"No features to store for {symbol}")
            return

        if "event_timestamp" not in features_df.columns:
            raise ValueError(
                "features_df must contain 'event_timestamp' column for feature store operations"
            )

        # Validate and clean event_timestamp column to prevent Dask/Feast issues
        if features_df["event_timestamp"].isnull().any():
            logger.warning(
                f"Found null values in event_timestamp for {symbol}, dropping rows with nulls"
            )
            return

        # Ensure event_timestamp is timezone-aware to prevent Feast timezone issues
        if features_df["event_timestamp"].dt.tz is None:
            features_df = features_df.copy()
            features_df["event_timestamp"] = features_df[
                "event_timestamp"
            ].dt.tz_localize("UTC")

        # Still create feature views even if no new data was stored
        # This is needed for online operations to work properly
        if not feature_service_metadata.feature_view_metadata_list:
            logger.warning(
                f"No feature view metadata provided for {symbol}, skipping Feast object creation"
            )
            return

        self._initialize_feast_objects(feature_service_metadata)

        # For backfill mode: filter to only requested period before storing
        # This removes warmup candles that were needed for computation but shouldn't be stored
        if (
            processing_context == "backfill"
            and requested_start_time is not None
            and requested_end_time is not None
        ):
            original_count = len(features_df)
            features_df = features_df[
                (features_df["event_timestamp"] >= requested_start_time)
                & (features_df["event_timestamp"] <= requested_end_time)
            ].copy()
            logger.info(
                f"Backfill mode: filtered features from {original_count} to {len(features_df)} records "
                f"for requested period [{requested_start_time} - {requested_end_time}]"
            )

        if features_df.empty:
            logger.warning(f"No features in requested period for {symbol}")
            return

        # Store features using appropriate strategy based on context
        if processing_context == "backfill":
            # Backfill: use batch mode (replace existing data for the period)
            logger.info(f"Storing {len(features_df)} features in BATCH mode (backfill)")
            stored_count = self.offline_repo.store_features_batch(
                [{"symbol": symbol, "features_df": features_df}]
            ).get(symbol, 0)
        else:
            # Training/Inference: use incremental mode (deduplicate)
            logger.info(
                f"Storing {len(features_df)} features in INCREMENTAL mode ({processing_context})"
            )
            stored_count = self.offline_repo.store_features_incrementally(
                features_df, symbol
            )

        if stored_count == 0:
            logger.warning(f"No new features stored for {symbol}")
            return

        logger.info(f"Stored {stored_count} feature records for {symbol}")

    def batch_materialize_features(
        self,
        features_df: DataFrame,
        symbol: str,
    ) -> None:
        """
        Materialize features for online serving (batch mode only).

        This method is designed for training/batch processing where large
        datasets are materialized from offline storage to online store.

        For inference mode, use push_features_to_online_store() instead
        to avoid unnecessary offline storage and improve performance.

        Args:
            features_df: DataFrame containing the computed features
            dataset_id: Identifier for the dataset to which these features belong
        """
        if "event_timestamp" not in features_df.columns:
            raise ValueError(
                "features_df must contain 'event_timestamp' column for materialization"
            )

        try:
            # Try materialization with timezone-aware timestamps
            start_date = features_df["event_timestamp"].min()
            end_date = features_df["event_timestamp"].max()

            # Ensure timestamps are timezone-aware to prevent pandas conversion issues
            if start_date.tz is None:
                start_date = start_date.tz_localize("UTC")
                end_date = end_date.tz_localize("UTC")

            # Get all feature views to materialize
            # This ensures we only materialize existing feature views
            feature_views = self.feature_store.list_feature_views()
            feature_view_names = [fv.name for fv in feature_views]

            if not feature_view_names:
                logger.warning(f"No feature views found to materialize for {symbol}")
                return

            logger.debug(f"Materializing feature views: {feature_view_names}")

            with pd.option_context("future.no_silent_downcasting", True):
                self.feature_store.materialize(
                    start_date=start_date,
                    end_date=end_date,
                    feature_views=feature_view_names,
                )
            logger.info(f"Materialized features for online serving: {symbol}")

        except (TypeError, AttributeError) as e:
            raise RuntimeError(f"Materialization failed for {symbol}: {e}") from e

    def push_features_to_online_store(
        self,
        features_df: DataFrame,
        symbol: str,
        feature_role: FeatureRoleEnum,
    ) -> None:
        """
        Push features directly to online store for real-time inference.

        This method bypasses offline storage and directly updates the online
        feature store, optimized for single-record inference scenarios.

        Use this method for:
        - Real-time inference where features are computed on-demand
        - Single-record updates that don't warrant offline storage
        - High-frequency scenarios where storage I/O would add latency

        Args:
            features_df: DataFrame containing the computed features (typically single record)
            dataset_id: Identifier for the dataset to which these features belong
        """
        if "event_timestamp" not in features_df.columns:
            raise ValueError(
                "features_df must contain 'event_timestamp' column for online store update"
            )

        if "symbol" not in features_df.columns:
            raise ValueError(
                "features_df must contain 'symbol' column for entity key extraction"
            )

        # Find all feature views for this role and symbol using tags
        try:
            all_feature_views = self.feature_store.list_feature_views(
                allow_cache=self.feast_provider.feature_store_config.cache_enabled
            )
            relevant_feature_views = [
                fv
                for fv in all_feature_views
                if fv.tags.get("feature_role") == feature_role.value
                and fv.tags.get("symbol") == symbol
            ]
        except Exception as e:
            raise RuntimeError(f"Failed to list feature views from store: {e}") from e

        if not relevant_feature_views:
            raise RuntimeError(
                f"No feature views found for role '{feature_role.value}' and symbol '{symbol}'. "
                f"Ensure offline features are stored first to create the feature views."
            )

        logger.debug(
            f"Found {len(relevant_feature_views)} feature views for role '{feature_role.value}' and symbol '{symbol}': "
            f"{[fv.name for fv in relevant_feature_views]}"
        )

        # Process each feature view individually
        total_pushed = 0
        for feature_view in relevant_feature_views:
            # Extract the field names from the feature view schema
            schema_field_names = {field.name for field in feature_view.schema}
            logger.debug(
                f"Feature view '{feature_view.name}' schema fields: {schema_field_names}"
            )

            # Filter DataFrame to only include columns for this specific feature view:
            # - event_timestamp (required by Feast)
            # - symbol (entity key)
            # - schema fields (actual feature columns for this view)
            required_columns = {"event_timestamp", "symbol"} | schema_field_names
            available_columns = set(features_df.columns)

            # Check if this feature view has any matching columns in the DataFrame
            matching_schema_fields = schema_field_names & available_columns
            if not matching_schema_fields:
                logger.debug(
                    f"Skipping feature view '{feature_view.name}' - no matching columns found. "
                    f"Required: {schema_field_names}, Available: {available_columns}"
                )
                continue

            # Filter to only include columns that exist in both the DataFrame and the schema
            columns_to_include = sorted(required_columns & available_columns)
            filtered_df = features_df[columns_to_include].copy()

            # Ensure entity column (symbol) is string type for Feast compatibility
            filtered_df["symbol"] = filtered_df["symbol"].astype(str)

            logger.debug(
                f"Filtered DataFrame for feature view '{feature_view.name}': "
                f"Original columns: {sorted(features_df.columns)}, "
                f"Filtered columns: {columns_to_include}"
            )

            # Push features directly to online store for this feature view
            self.feature_store.write_to_online_store(
                feature_view_name=feature_view.name,
                df=filtered_df,
            )
            total_pushed += len(filtered_df)
            logger.debug(
                f"Pushed {len(filtered_df)} feature records to online store for feature view '{feature_view.name}'"
            )

        logger.info(
            f"Successfully pushed features to {len(relevant_feature_views)} feature views "
            f"for role '{feature_role.value}' and symbol '{symbol}'. Total records: {total_pushed}"
        )

    def _initialize_feast_objects(
        self, feature_service_metadata: FeatureServiceMetadata
    ) -> None:
        """
        Create and apply Feast feature views and services.

        Args:
            feature_service_metadata: Feature service metadata containing all necessary information
        """
        if (
            feature_service_metadata.feature_view_metadata_list is None
            or len(feature_service_metadata.feature_view_metadata_list) == 0
        ):
            logger.warning(
                f"No feature view requests provided for {feature_service_metadata.dataset_identifier.symbol}, skipping creation"
            )
            return

        service_name = get_feature_service_name(request=feature_service_metadata)

        # Create feature service combining both views
        self.feast_provider.get_or_create_feature_service(
            service_name=service_name,
            feature_view_requests=feature_service_metadata.feature_view_metadata_list,
        )
