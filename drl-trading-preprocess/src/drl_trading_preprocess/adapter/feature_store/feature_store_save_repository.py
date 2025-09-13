import logging

from drl_trading_adapter.adapter.feature_store.provider import FeastProvider
from injector import inject
from pandas import DataFrame

from drl_trading_adapter.adapter.feature_store.util.feature_store_utilities import get_feature_service_name
from drl_trading_common.enum import FeatureRoleEnum
from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo
from drl_trading_core.common.model import FeatureViewRequestContainer
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
        features_df: DataFrame,
        symbol: str,
        feature_version_info: FeatureConfigVersionInfo,
        feature_view_requests: list[FeatureViewRequestContainer],
    ) -> None:
        """
        Store computed features using the configured offline repository and
        create Feast feature views and services.

        Args:
            features_df: DataFrame containing the computed features
            dataset_id: Identifier for the dataset
            feature_version_info: Version information for feature tracking
        """
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

        # Store features using the configured offline repository
        stored_count = self.offline_repo.store_features_incrementally(
            features_df, symbol
        )

        if stored_count == 0:
            logger.warning(f"No new features stored for {symbol}")
            # Still create feature views even if no new data was stored
            # This is needed for online operations to work properly
            self._create_or_update_features(symbol, feature_version_info, feature_view_requests)
            return

        logger.info(f"Stored {stored_count} feature records for {symbol}")
        # Create and apply Feast feature views
        self._create_or_update_features(symbol, feature_version_info, feature_view_requests)

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

        feature_view_name = self.feature_view_name_mapper.map(feature_role)

        # Get the feature view to determine which columns should be included
        try:
            feature_view = self.feature_store.get_feature_view(feature_view_name)
        except Exception as e:
            raise RuntimeError(
                f"Feature view '{feature_view_name}' not found. "
                f"Ensure offline features are stored first to create the feature view. Error: {e}"
            ) from e

        # Extract the field names from the feature view schema
        schema_field_names = {field.name for field in feature_view.schema}
        logger.debug(
            f"Feature view '{feature_view_name}' schema fields: {schema_field_names}"
        )

        # Filter DataFrame to only include required columns:
        # - event_timestamp (required by Feast)
        # - symbol (entity key)
        # - schema fields (actual feature columns)
        required_columns = {"event_timestamp", "symbol"} | schema_field_names
        available_columns = set(features_df.columns)

        # Find missing schema fields
        missing_fields = schema_field_names - available_columns
        if missing_fields:
            raise ValueError(
                f"Features DataFrame is missing required schema fields: {missing_fields}. "
                f"Available columns: {available_columns}"
            )

        # Filter to only include columns that exist in both the DataFrame and the schema
        columns_to_include = sorted(required_columns & available_columns)
        filtered_df = features_df[columns_to_include].copy()

        # Ensure entity column (symbol) is string type for Feast compatibility
        filtered_df["symbol"] = filtered_df["symbol"].astype(str)

        logger.debug(
            f"Filtered DataFrame for online store push: "
            f"Original columns: {sorted(features_df.columns)}, "
            f"Filtered columns: {columns_to_include}"
        )

        logger.debug(f"Symbol values before push: {filtered_df['symbol'].tolist()}")
        logger.debug(f"Symbol dtype: {filtered_df['symbol'].dtype}")

        # Push features directly to online store without offline storage
        # This avoids creating many small parquet files during inference
        # Use Feast's write_to_online_store for direct online updates
        self.feature_store.write_to_online_store(
            feature_view_name=feature_view_name,
            df=filtered_df,
        )
        logger.debug(
            f"Pushed {len(filtered_df)} feature records of feature-view {feature_view_name} to online store: {symbol}"
        )

    def _create_or_update_features(
        self,
        symbol: str,
        feature_version_info: FeatureConfigVersionInfo,
        feature_view_requests: list[FeatureViewRequestContainer],
    ) -> None:
        """
        Create and apply Feast feature views and services.

        Args:
            symbol: Symbol for which to create feature views
            feature_version_info: Version info for feature tracking
            feature_view_requests: List of feature view requests to create
        """

        if feature_view_requests is None or len(feature_view_requests) == 0:
            logger.warning(
                f"No feature view requests provided for {symbol}, skipping creation"
            )
            return

        feature_service_role = feature_view_requests[0].feature_role

        service_name = get_feature_service_name(
            feature_service_role=feature_service_role,
            symbol=symbol,
            feature_version_info=feature_version_info,
        )

        # Create feature service combining both views
        self.feast_provider.get_or_create_feature_service(
            service_name=service_name,
            feature_view_requests=feature_view_requests,
        )
