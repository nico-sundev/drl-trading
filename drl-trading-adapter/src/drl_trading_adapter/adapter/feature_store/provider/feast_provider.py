import logging
from datetime import timedelta

from feast import (
    Entity,
    FeatureService,
    FeatureStore,
    FeatureView,
    Field,
    FileSource,
    OnDemandFeatureView,
    ValueType,
)
from feast.data_format import ParquetFormat
from feast.types import String
from injector import inject

from drl_trading_adapter.adapter.feature_store.offline import IOfflineFeatureRepository
from drl_trading_adapter.adapter.feature_store.provider.feature_store_wrapper import (
    FeatureStoreWrapper,
)
from drl_trading_adapter.adapter.feature_store.provider.mapper.feature_field_mapper import (
    IFeatureFieldFactory,
)
from drl_trading_adapter.adapter.feature_store.util.feature_store_utilities import get_feature_view_name
from drl_trading_core.core.model.feature.feature_metadata import FeatureMetadata
from drl_trading_common.config import FeatureStoreConfig
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_core.core.dto.feature_view_metadata import FeatureViewMetadata

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
        feature_field_mapper: IFeatureFieldFactory,
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

    def _process_feature_view_creation_requests(
        self, requests: list[FeatureViewMetadata]
    ) -> list[FeatureView]:
        """
        Create feature views using request containers.

        Args:
            requests: list of featureview Container with all required parameters for feature view creation

        Returns:
            FeatureView: The created feature views

        Raises:
            ValueError: If any parameter is invalid or missing required attributes
            OSError: If file system operations fail
            Exception: For unexpected Feast-related errors during feature view creation
        """
        created_feature_views: list[FeatureView] = []

        try:
            for request in requests:
                feature_view_name = request.feature_metadata.__str__()

                # Validate request parameters
                request.validate()

                # Validate config
                self._validate_feature_store_config()

                # Create feature view with validated and sanitized inputs
                created_feature_views.append(
                    self._handle_feature_view_creation(request)
                )

        except (ValueError, RuntimeError, OSError) as e:
            # Re-raise known exception types with context
            logger.error(
                f"Error creating feature view '{feature_view_name}' for symbol '{request.dataset_identifier.symbol}': {e}"
            )
            raise
        except Exception as e:
            # Catch any unexpected errors and wrap them
            logger.error(
                f"Unexpected error creating feature view '{feature_view_name}' for symbol '{request.dataset_identifier.symbol}': {e}"
            )
            raise RuntimeError(
                f"Unexpected error during feature view creation: {e}"
            ) from e

        # Batch apply all created feature views to the store
        self.feature_store.apply([*created_feature_views])
        return created_feature_views

    def _validate_feature_store_config(self) -> None:
        """
        Validate that the feature store configuration is valid.

        Raises:
            RuntimeError: If feature store is disabled or in invalid state
        """

        # Validate TTL configuration
        if (
            not hasattr(self.feature_store_config, "ttl_days")
            or self.feature_store_config.ttl_days <= 0
        ):
            raise ValueError("Feature store config must have a positive ttl_days value")

    def _handle_feature_view_creation(
        self, request: FeatureViewMetadata
    ) -> FeatureView:
        """
        Internal method to create the feature view with validated parameters.

        Args:
            request: Validated feature view request

        Returns:
            FeatureView: The created feature view
        """
        # Get sanitized inputs
        symbol = request.get_sanitized_symbol()
        base_feature_view_name = request.feature_metadata.__str__()

        # Create symbol-specific feature view name for proper isolation
        feature_view_name = get_feature_view_name(
            base_feature_view_name=base_feature_view_name,
            request=request
        )

        # Create data directory and file source
        source = self._create_file_source(feature_view_name, symbol)

        fields = self._create_fields_from_features([request.feature_metadata])

        # Add symbol field to satisfy Feast entity join key requirement
        # TODO: Consider making join keys configurable in the future
        fields.append(Field(name="symbol", dtype=String))

        # Create entity
        entity = self._get_or_create_entity()

        # Create and return the feature view
        logger.info(
            f"Creating feature view '{feature_view_name}' for symbol '{symbol}', timeframe '{request.dataset_identifier.timeframe.value}', role '{request.feature_metadata.feature_role.value}'"
        )
        fv = self._create_feature_view(
            feature_view_name=feature_view_name,
            entity=entity,
            fields=fields,
            source=source,
            feature_role=request.feature_metadata.feature_role,
            symbol=symbol,
        )
        return fv

    def _create_file_source(
        self,
        feature_view_name: str,
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
                path=offline_store_path,
                s3_endpoint_override= self.feature_store_config.s3_repo_config.endpoint_url if self.feature_store_config.s3_repo_config else None,
                file_format=ParquetFormat(),
                timestamp_field="event_timestamp",
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create FileSource for feature view '{feature_view_name}': {e}"
            ) from e

    def _create_fields_from_features(
        self, features_metadata: list[FeatureMetadata]
    ) -> list[Field]:
        """Create Feast fields from feature metadata."""
        fields = []

        try:
            for feature_metadata in features_metadata:
                feature_fields = self.feature_field_mapper.create_fields(feature_metadata)
                if feature_fields:
                    fields.extend(feature_fields)
        except Exception as e:
            raise RuntimeError(f"Failed to create fields for features: {e}") from e

        return fields

    def _get_or_create_entity(self) -> Entity:
        """
        Get existing entity or create a shared entity for all symbols.

        Returns:
            Entity: Shared Feast entity for all trading symbols
        """
        # Use single shared entity name for all symbols
        entity_name = self.feature_store_config.entity_name

        try:
            return self.feature_store.get_entity(
                name=entity_name,
                allow_registry_cache=self.feature_store_config.cache_enabled
            )
        except Exception as e:
            logger.debug(
                f"Entity '{entity_name}' not found, creating new: {e}"
            )
            logger.info("Creating shared entity for all symbols")
            entity = self._create_entity()
            self.feature_store.apply([entity])
            return entity

    def _create_feature_view(
        self,
        feature_view_name: str,
        entity: Entity,
        fields: list[Field],
        source: FileSource,
        feature_role: FeatureRoleEnum,
        symbol: str,
    ) -> FeatureView:
        """Create the final Feast FeatureView with error handling."""
        try:
            return FeatureView(
                name=feature_view_name,
                entities=[entity],
                ttl=timedelta(days=self.feature_store_config.ttl_days),
                schema=fields,
                # online=self.feature_store_config.online_enabled,  # Use config setting
                online=True,  # Use config setting,
                # offline=True,  # Use config setting,
                source=source,
                tags={
                    "feature_role": f"{feature_role.value}",
                    "symbol": symbol
                },
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create FeatureView '{feature_view_name}': {e}"
            ) from e

    def _get_or_create_feature_views(
        self, requests: list[FeatureViewMetadata]
    ) -> list[FeatureView | OnDemandFeatureView]:
        """
        Get existing feature views or create new ones if not yet existing.

        Args:
            requests: List of feature view requests to process

        Returns:
            list[FeatureView | OnDemandFeatureView]: Existing or newly created feature views
        """

        if len(requests) == 0:
            logger.warning(
                "No feature view requests provided for _get_or_create_feature_views."
            )
            return []

        feature_views: list[FeatureView | OnDemandFeatureView] = []

        # Create a dictionary for quick feature name lookup
        feature_requests_map: dict[str, FeatureViewMetadata] = {
            request.feature_metadata.__str__(): request
            for request in requests
        }

        # Attempt to fetch existing feature views
        symbol = requests[0].get_sanitized_symbol()
        role = requests[0].feature_metadata.feature_role.value
        existing_feature_views = self._find_feature_views_by_name(
            list(feature_requests_map.keys()), symbol=symbol, feature_role=role
        )
        feature_views.extend(existing_feature_views)

        # Remove found feature views from the requests map
        for fv in existing_feature_views:
            feature_requests_map.pop(fv.name, None)

        # Create any remaining feature views that were not found
        feature_views.extend(
            self._process_feature_view_creation_requests(
                list(feature_requests_map.values())
            )
        )

        return feature_views

    def _find_feature_views_by_name(
        self, feature_view_names: list[str], **kwargs: str
    ) -> list[FeatureView]:
        """
        Find feature views by name.

        Args:
            name: Name of the feature view to search for

        Returns:
            list[FeatureView]: List of matching feature views
        """
        try:
            all_feature_views = self.feature_store.list_feature_views(
                allow_cache=self.feature_store_config.cache_enabled, tags=kwargs
            )
            matching_fvs = [
                fv for fv in all_feature_views if fv.name in feature_view_names
            ]
            return matching_fvs
        except Exception as e:
            raise RuntimeError(f"Failed to list feature views from store: {e}") from e

    def get_feature_service(self, service_name: str) -> FeatureService:
        """
        Get an existing feature service by name.

        Args:
            service_name: Name of the feature service to retrieve

        Returns:
            FeatureService: The requested feature service
        """
        try:
            return self.feature_store.get_feature_service(
                name=service_name,
                allow_cache=self.feature_store_config.cache_enabled
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to get feature service '{service_name}': {e}"
            ) from e

    def get_or_create_feature_service(
        self,
        service_name: str,
        feature_view_requests: list[FeatureViewMetadata],
    ) -> FeatureService:
        """
        Create a symbol-specific feature service based on feature description.

        Args:
            symbol: Trading symbol for this service
            description: Description of the feature service
            feature_views: List of feature views to include in the service

        Returns:
            FeatureService: The created or existing feature service
        """
        try:
            return self.get_feature_service(service_name=service_name)
        except Exception as e:
            logger.debug(
                f"Feature service not found for {service_name}, creating new service: {e}"
            )

        return self._create_feature_service(
            service_name=service_name,
            feature_view_requests=feature_view_requests,
        )

    def _create_feature_service(
        self,
        service_name: str,
        feature_view_requests: list[FeatureViewMetadata],
    ) -> FeatureService:
        feature_views: list[FeatureView | OnDemandFeatureView] = (
            self._get_or_create_feature_views(requests=feature_view_requests)
        )

        if len(feature_views) == 0:
            raise ValueError(
                f"No feature views available to create feature service '{service_name}'"
            )

        logger.info(
            f"Creating feature service '{service_name}' with {len(feature_views)} feature views"
        )
        fs = FeatureService(name=service_name, features=feature_views)
        self.feature_store.apply([fs])
        return fs

    def _create_entity(self) -> Entity:
        """
        Create a shared entity for all trading symbols.

        According to Feast best practices, entities should be reused across
        feature views. The entity uses 'symbol' as the join key, allowing
        different symbols to be isolated by their data values while sharing
        the same entity definition.

        Args:

        Returns:
            Entity: Shared Feast entity for all trading symbols
        """
        # Create shared entity name for all symbols
        entity_name = self.feature_store_config.entity_name

        return Entity(
            name=entity_name,
            join_keys=["symbol"],  # Column name in DataFrame, not the symbol value
            description="Shared entity for all trading symbol asset price data",
            value_type=ValueType.STRING
        )
