import logging
from datetime import timedelta
from typing import Optional

from feast import (
    Entity,
    FeatureService,
    FeatureStore,
    FeatureView,
    Field,
    FileSource,
    OnDemandFeatureView,
)
from injector import inject

from .feature_store_wrapper import FeatureStoreWrapper
from .mapper.feature_field_mapper import IFeatureFieldMapper
from drl_trading_adapter.adapter.feature_store.offline import IOfflineFeatureRepository
from drl_trading_common.base import BaseFeature
from drl_trading_common.config import FeatureStoreConfig
from drl_trading_common.model import (
    FeatureConfigVersionInfo,
)
from drl_trading_core.common.model import FeatureViewRequest

logger = logging.getLogger(__name__)


@inject
class FeastProvider:
    """
    A class to provide access to Feast features.
    """

    def __init__(
        self,
        feature_store_config: FeatureStoreConfig,
        feature_store_wrapper: FeatureStoreWrapper,
        offline_feature_repo: IOfflineFeatureRepository,
        feature_field_mapper: IFeatureFieldMapper,
    ):
        self.feature_store_config = feature_store_config
        self.feature_store = feature_store_wrapper.get_feature_store()
        self.offline_feature_repo = offline_feature_repo
        self.feature_field_mapper = feature_field_mapper

    def get_feature_store(self) -> FeatureStore:
        """
        Getter for the feature_store instance.

        Returns:
            FeatureStore: The Feast FeatureStore instance.
        """
        return self.feature_store

    def get_offline_repo(self) -> IOfflineFeatureRepository:
        """
        Getter for the offline_feature_repo instance.

        Returns:
            IOfflineFeatureRepository: The offline feature repository instance.
        """
        return self.offline_feature_repo

    def is_enabled(self) -> bool:
        """
        Check if the feature store is enabled.

        Returns:
            bool: True if the feature store is enabled, False otherwise
        """
        return bool(self.feature_store_config.enabled)

    def create_feature_view_from_request(
        self, request: FeatureViewRequest
    ) -> FeatureView:
        """
        Create a feature view using a request container for better parameter management.

        This is the preferred method for new code as it provides:
        - Better parameter grouping and validation
        - Improved readability at call sites
        - Easier testing and mocking

        Args:
            request: Container with all required parameters for feature view creation

        Returns:
            FeatureView: The created feature view

        Raises:
            ValueError: If any parameter is invalid or missing required attributes
            RuntimeError: If feature store is disabled or in invalid state
            OSError: If file system operations fail
            Exception: For unexpected Feast-related errors during feature view creation
        """
        try:
            # Validate request parameters
            request.validate()

            # Validate system state
            self._validate_feature_store_state()

            # Create feature view with validated and sanitized inputs
            return self._create_feature_view_internal(request)

        except (ValueError, RuntimeError, OSError) as e:
            # Re-raise known exception types with context
            logger.error(
                f"Error creating feature view '{request.feature_view_name}' for symbol '{request.symbol}': {e}"
            )
            raise
        except Exception as e:
            # Catch any unexpected errors and wrap them
            logger.error(
                f"Unexpected error creating feature view '{request.feature_view_name}' for symbol '{request.symbol}': {e}"
            )
            raise RuntimeError(
                f"Unexpected error during feature view creation: {e}"
            ) from e

    def _validate_feature_store_state(self) -> None:
        """
        Validate that the feature store is in a valid state for operations.

        Raises:
            RuntimeError: If feature store is disabled or in invalid state
        """
        if not self.is_enabled():
            raise RuntimeError(
                "Feature store is disabled - cannot create feature views"
            )

        if self.feature_store is None:
            raise RuntimeError("Feature store is not initialized")

        # Validate that offline repository is available
        if self.offline_feature_repo is None:
            raise RuntimeError("Offline feature repository is not configured")

        # Validate TTL configuration
        if (
            not hasattr(self.feature_store_config, "ttl_days")
            or self.feature_store_config.ttl_days <= 0
        ):
            raise ValueError("Feature store config must have a positive ttl_days value")

    def _create_feature_view_internal(self, request: FeatureViewRequest) -> FeatureView:
        """
        Internal method to create the feature view with validated parameters.

        Args:
            request: Validated feature view request

        Returns:
            FeatureView: The created feature view
        """
        # Get sanitized inputs
        symbol = request.get_sanitized_symbol()
        feature_view_name = request.get_sanitized_feature_view_name()

        # Create data directory and file source
        source = self._create_file_source(
            feature_view_name, request.feature_version_info, symbol
        )

        # Get features from the request and create fields
        features_for_role = self._get_features_from_request(request)
        fields = self._create_fields_for_features(features_for_role)

        # Create entity
        entity = self._create_entity_with_error_handling(symbol)

        # Create and return the feature view
        return self._create_feast_feature_view(
            feature_view_name=feature_view_name,
            entity=entity,
            fields=fields,
            source=source,
            symbol=symbol,
        )

    def _create_file_source(
        self,
        feature_view_name: str,
        feature_version_info: FeatureConfigVersionInfo,
        symbol: str,
    ) -> FileSource:
        """Create file source with error handling."""
        # Delegate path resolution to the offline repository implementation
        try:
            offline_store_path = self.offline_feature_repo.get_repo_path(symbol)
        except Exception as e:
            raise RuntimeError(
                f"Failed to get repository path for symbol '{symbol}': {e}"
            ) from e

        # Create file source with error handling
        try:
            return FileSource(
                name=f"view_{feature_view_name}_v{feature_version_info.semver}-{feature_version_info.hash}",
                path=offline_store_path,
                timestamp_field="event_timestamp",
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create FileSource for feature view '{feature_view_name}': {e}"
            ) from e

    def _get_features_from_request(self, request: FeatureViewRequest) -> list[BaseFeature]:
        """Get features from the FeatureViewRequest with validation."""

        features_from_request = request.features if hasattr(request, 'features') else []

        if not features_from_request:
            logger.warning(
                f"No features found in request for role {request.get_role_description()}, creating empty feature view"
            )

        logger.debug(
            f"Feast feature view will be created for feature role: {request.get_role_description()}"
        )
        return features_from_request

    def _create_fields_for_features(
        self, features_for_role: list[BaseFeature]
    ) -> list[Field]:
        """Create Feast fields from features with error handling."""
        fields = []

        try:
            for feature in features_for_role:
                feature_fields = self.feature_field_mapper.create_fields(feature)
                if feature_fields:
                    fields.extend(feature_fields)
        except Exception as e:
            raise RuntimeError(f"Failed to create fields for features: {e}") from e

        return fields

    def _create_entity_with_error_handling(self, symbol: str) -> Entity:
        """Create entity with error handling."""
        try:
            return self.get_entity(symbol)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create entity for symbol '{symbol}': {e}"
            ) from e

    def _create_feast_feature_view(
        self,
        feature_view_name: str,
        entity: Entity,
        fields: list[Field],
        source: FileSource,
        symbol: str,
    ) -> FeatureView:
        """Create the final Feast FeatureView with error handling."""
        try:
            return FeatureView(
                name=feature_view_name,
                entities=[entity],
                ttl=timedelta(days=self.feature_store_config.ttl_days),
                schema=fields,
                online=self.feature_store_config.online_enabled,  # Use config setting
                source=source,
                tags={
                    "symbol": symbol,
                },
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create FeatureView '{feature_view_name}': {e}"
            ) from e

    def create_feature_service(
        self,
        symbol: str,
        feature_version_info: FeatureConfigVersionInfo,
        feature_views: Optional[list[FeatureView | OnDemandFeatureView]] = None,
    ) -> FeatureService:

        try:
            existing_service = self.feature_store.get_feature_service(
                name=f"service_{symbol}_v{feature_version_info.semver}-{feature_version_info.hash}"
            )
            if existing_service:
                logger.debug(
                    f"Feature service already exists for symbol {symbol}, reusing existing service"
                )
                return existing_service
        except Exception as e:
            logger.debug(
                f"Feature service not found for symbol {symbol}, creating new service: {e}"
            )

        if feature_views is None:
            feature_views = []

        return FeatureService(
            name=f"service_{symbol}_v{feature_version_info.semver}-{feature_version_info.hash}",
            features=feature_views,
        )

    def get_entity(self, symbol: str) -> Entity:
        """
        Create an entity for the given dataset identifier.

        Args:
            dataset_id: Identifier for the dataset containing symbol and timeframe

        Returns:
            Entity: Feast entity for this symbol/timeframe combination
        """
        return Entity(
            name=self.feature_store_config.entity_name,
            join_keys=["symbol"],  # Column name in DataFrame, not the symbol value
            description=f"Entity for {symbol} asset price data",
        )
