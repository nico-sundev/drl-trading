import logging
import os
from datetime import timedelta
from typing import Optional

from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_common.config.feature_config import FeatureStoreConfig
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.model.feature_config_version_info import (
    FeatureConfigVersionInfo,
)
from feast import (
    Entity,
    FeatureService,
    FeatureStore,
    FeatureView,
    Field,
    FileSource,
    OnDemandFeatureView,
)
from feast.types import Float32

from drl_trading_core.preprocess.feature.feature_manager import FeatureManager

logger = logging.getLogger(__name__)


class FeastProvider:
    """
    A class to provide access to Feast features.
    """

    def __init__(
        self,
        feature_store_config: FeatureStoreConfig,
        feature_manager: FeatureManager,
    ):
        """
        Initializes the FeastProvider with the given project, registry, and online store.

        :param project: The name of the Feast project.
        :param registry: The path to the Feast registry.
        :param online_store: The path to the Feast online store.
        """
        self.feature_manager = feature_manager
        self.feature_store_config = feature_store_config
        self._feature_store = FeatureStore(repo_path=self._resolve_feature_store_path())

    def get_feature_store(self) -> FeatureStore:
        """
        Getter for the feature_store instance.

        Returns:
            FeatureStore: The Feast FeatureStore instance.
        """
        return self._feature_store

    def is_enabled(self) -> bool:
        """
        Check if the feature store is enabled.

        Returns:
            bool: True if the feature store is enabled, False otherwise
        """
        return self.feature_store_config.enabled

    def _resolve_feature_store_path(self) -> Optional[str]:
        """
        Resolve the feature store path based on configuration.
        If the path is relative, it will be resolved against the project root directory.

        Returns:
            Optional[str]: Absolute path to the feature store repository, or None if not enabled
        """
        if not self.feature_store_config.enabled:
            return None

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        )
        if not os.path.isabs(self.feature_store_config.repo_path):
            abs_file_path = os.path.join(
                project_root, self.feature_store_config.repo_path
            )
        else:
            abs_file_path = self.feature_store_config.repo_path
        return abs_file_path

    def _create_fields(self, feature: BaseFeature) -> list[Field]:
        """
        Create fields for the feature view based on the feature's type and role.

        Args:
            feature: The feature for which fields are created

        Returns:
            list[Field]: List of fields for the feature view
        """
        fields = []
        feature_name = self._get_feature_name(
            feature.get_feature_name(), feature.get_config()
        )
        logger.debug(f"Feast fields will be created for feature: {feature_name}")

        for sub_feature in feature.get_sub_features_names():
            feast_field_name = f"{feature_name}_{sub_feature}"
            logger.debug(f"Creating feast field:{feast_field_name}")
            fields.append(Field(name=feast_field_name, dtype=Float32))

        return fields

    def create_feature_view(
        self,
        symbol: str,
        feature_view_name: str,
        feature_role: FeatureRoleEnum,
        feature_version_info: FeatureConfigVersionInfo,
    ) -> FeatureView:
        """
        Create a feature view for the given feature parameters.

        Args:
            symbol: The symbol for which the feature view is created
            feature_view_name: The name of the feature view
            feature_role: The role of the feature
            feature_version_info: Version information for the feature configuration

        Returns:
            FeatureView: The created feature view
        """

        # Create a file source for the feature
        source = FileSource(
            name=f"view_{feature_view_name}_v{feature_version_info.semver}-{feature_version_info.hash}",
            path=self.feature_store_config.offline_store_path,
            timestamp_field="event_timestamp",
        )

        # Create fields for the feature view
        fields = []

        logger.debug(
            f"Feast feature view will be created for feature role: {feature_role.value}"
        )

        for feature in self.feature_manager.get_features_by_role(feature_role):
            fields.extend(self._create_fields(feature))

        # Create and return the feature view
        return FeatureView(
            name=feature_view_name,
            entities=[self.get_entity(symbol)],
            ttl=timedelta(days=self.feature_store_config.ttl_days),
            schema=fields,
            online=False,
            source=source,
            tags={
                "symbol": symbol,
            },
        )

    def create_feature_service(
        self,
        symbol: str,
        feature_version_info: FeatureConfigVersionInfo,
        feature_views: Optional[list[FeatureView | OnDemandFeatureView]] = None,
    ):

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
            join_keys=[symbol],
            description=f"Entity for {symbol} asset price data",
        )

    def _get_feature_name(
        self, feature_name: str, feature_config: BaseParameterSetConfig
    ) -> str:
        """
        Create a unique feature name based on the feature name and its config hash.

        Args:
            feature_name: Name of the feature
            feature_config: Configuration of the feature

        Returns:
            str: A unique name for the feature
        """
        return f"{feature_name}_{feature_config.hash_id()}"
