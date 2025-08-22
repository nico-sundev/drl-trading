from __future__ import annotations
import logging
from datetime import timedelta
from typing import Optional

from drl_trading_adapter.adapter.feature_store.feast.feature_store_wrapper import FeatureStoreWrapper
from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.config.feature_config import FeatureStoreConfig
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo
from feast import Entity, FeatureService, FeatureStore, FeatureView, Field, FileSource, OnDemandFeatureView
from feast.types import Float32
from injector import inject

from drl_trading_core.common.model.feature_view_request import FeatureViewRequest
from drl_trading_core.preprocess.feature.feature_manager import FeatureManager
from drl_trading_core.preprocess.feature_store.port.feature_store_provider_port import (
    IFeatureStoreProvider,
)
from drl_trading_core.preprocess.feature_store.port.offline_feature_repo_interface import IOfflineFeatureRepository

logger = logging.getLogger(__name__)


@inject
class FeastProvider(IFeatureStoreProvider):
    """Concrete adapter implementation of IFeatureStoreProvider.

    Responsible for orchestrating Feast entities, feature views, and feature services.
    Depends on injected offline repository for path resolution to maintain hexagonal
    boundaries (core unaware of filesystem/S3 specifics).
    """

    def __init__(
        self,
        feature_store_config: FeatureStoreConfig,
        feature_manager: FeatureManager,
        feature_store_wrapper: FeatureStoreWrapper,
        offline_feature_repo: IOfflineFeatureRepository,
    ) -> None:
        self.feature_manager = feature_manager
        self.feature_store_config = feature_store_config
        self.feature_store = feature_store_wrapper.get_feature_store()
        self._offline_repo = offline_feature_repo

    def get_feature_store(self) -> FeatureStore:  # type: ignore[override]
        return self.feature_store  # type: ignore[return-value]

    def is_enabled(self) -> bool:  # type: ignore[override]
        return bool(self.feature_store_config.enabled)

    def create_feature_view(self, symbol: str, feature_view_name: str, feature_role: FeatureRoleEnum, feature_version_info: FeatureConfigVersionInfo) -> FeatureView:
        request = FeatureViewRequest.create(
            symbol=symbol,
            feature_view_name=feature_view_name,
            feature_role=feature_role,
            feature_version_info=feature_version_info,
        )
        return self.create_feature_view_from_request(request)

    def create_feature_view_from_request(self, request: FeatureViewRequest) -> FeatureView:
        try:
            request.validate()
            self._validate_feature_store_state()
            return self._create_feature_view_internal(request)
        except (ValueError, RuntimeError, OSError) as e:
            logger.error(
                f"Error creating feature view '{request.feature_view_name}' for symbol '{request.symbol}': {e}"
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error creating feature view '{request.feature_view_name}' for symbol '{request.symbol}': {e}"
            )
            raise RuntimeError(f"Unexpected error during feature view creation: {e}") from e

    def _validate_feature_store_state(self) -> None:
        if not self.is_enabled():
            raise RuntimeError("Feature store is disabled - cannot create feature views")
        if self.feature_store is None:
            raise RuntimeError("Feature store is not initialized")
        if not hasattr(self.feature_store_config, "ttl_days") or self.feature_store_config.ttl_days <= 0:
            raise ValueError("Feature store config must have a positive ttl_days value")

    def _create_feature_view_internal(self, request: FeatureViewRequest) -> FeatureView:
        symbol = request.get_sanitized_symbol()
        feature_view_name = request.get_sanitized_feature_view_name()
        source = self._create_file_source(feature_view_name, request.feature_version_info, symbol)
        features_for_role = self._get_features_for_role(request.feature_role, request.get_role_description())
        fields = self._create_fields_for_features(features_for_role)
        entity = self._create_entity_with_error_handling(symbol)
        return self._create_feast_feature_view(
            feature_view_name=feature_view_name,
            entity=entity,
            fields=fields,
            source=source,
            symbol=symbol,
        )

    def _create_file_source(self, feature_view_name: str, feature_version_info: FeatureConfigVersionInfo, symbol: str) -> FileSource:
        offline_store_path = self._offline_repo.get_repo_path(symbol)
        return FileSource(
            name=f"view_{feature_view_name}_v{feature_version_info.semver}-{feature_version_info.hash}",
            path=offline_store_path,
            timestamp_field="event_timestamp",
        )

    def _get_features_for_role(self, feature_role: Optional[FeatureRoleEnum], role_description: str) -> list[BaseFeature]:
        features_for_role: list[BaseFeature] = []
        if feature_role is not None:
            # FeatureManager.get_features_by_role returns a list; extend to avoid nested lists
            features = self.feature_manager.get_features_by_role(feature_role)
            if features:
                features_for_role.extend(features)
        if not features_for_role:
            logger.warning(f"No features found for role {role_description}, creating empty feature view")
        logger.debug(f"Feast feature view will be created for feature role: {role_description}")
        return features_for_role

    def _create_fields_for_features(self, features_for_role: list[BaseFeature]) -> list[Field]:
        fields: list[Field] = []
        for feature in features_for_role:
            if feature is None:
                logger.warning("Skipping None feature in feature list")
                continue
            feature_fields = self._create_fields(feature)
            if feature_fields:
                fields.extend(feature_fields)
        return fields

    def _create_entity_with_error_handling(self, symbol: str) -> Entity:
        try:
            return self.get_entity(symbol)
        except Exception as e:
            raise RuntimeError(f"Failed to create entity for symbol '{symbol}': {e}") from e

    def _create_feast_feature_view(self, feature_view_name: str, entity: Entity, fields: list[Field], source: FileSource, symbol: str) -> FeatureView:
        return FeatureView(
            name=feature_view_name,
            entities=[entity],
            ttl=timedelta(days=self.feature_store_config.ttl_days),
            schema=fields,
            online=self.feature_store_config.online_enabled,
            source=source,
            tags={"symbol": symbol},
        )

    def create_feature_service(self, symbol: str, feature_version_info: FeatureConfigVersionInfo, feature_views: Optional[list[FeatureView | OnDemandFeatureView]] = None) -> FeatureService:
        try:
            existing_service = self.feature_store.get_feature_service(
                name=f"service_{symbol}_v{feature_version_info.semver}-{feature_version_info.hash}"
            )
            if existing_service:
                logger.debug(f"Feature service already exists for symbol {symbol}, reusing existing service")
                return existing_service
        except Exception as e:
            logger.debug(f"Feature service not found for symbol {symbol}, creating new service: {e}")
        if feature_views is None:
            feature_views = []
        return FeatureService(
            name=f"service_{symbol}_v{feature_version_info.semver}-{feature_version_info.hash}",
            features=feature_views,
        )

    def get_entity(self, symbol: str) -> Entity:
        return Entity(
            name=self.feature_store_config.entity_name,
            join_keys=["symbol"],
            description=f"Entity for {symbol} asset price data",
        )

    def _get_field_base_name(self, feature: BaseFeature) -> str:
        config = feature.get_config()
        config_string = f"_{feature.get_config_to_string()}_{config.hash_id()}" if config else ""
        return f"{feature.get_feature_name()}{config_string}"

    def _create_fields(self, feature: BaseFeature) -> list[Field]:
        feature_name = self._get_field_base_name(feature)
        logger.debug(f"Feast fields will be created for feature: {feature_name}")
        if len(feature.get_sub_features_names()) == 0:
            logger.debug(f"Creating feast field:{feature_name}")
            return [Field(name=feature_name, dtype=Float32)]
        fields: list[Field] = []
        for sub_feature in feature.get_sub_features_names():
            feast_field_name = f"{feature_name}_{sub_feature}"
            logger.debug(f"Creating feast field:{feast_field_name}")
            fields.append(Field(name=feast_field_name, dtype=Float32))
        return fields
