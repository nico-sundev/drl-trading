import logging
import os
from datetime import timedelta
from typing import Optional

from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_common.config.feature_config import FeatureStoreConfig
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.model.dataset_identifier import DatasetIdentifier
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
        dataset_id: DatasetIdentifier,
        feature_view_name: str,
        feature_role: FeatureRoleEnum,
        feature_version_info: FeatureConfigVersionInfo,
    ) -> FeatureView:
        """
        Create a feature view for the given feature parameters.

        Args:
            entity: The entity for which the feature view is created
            dataset_id: Identifier for the dataset containing symbol and timeframe

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
            entities=[self.get_entity(dataset_id)],
            ttl=timedelta(days=self.feature_store_config.ttl_days),
            schema=fields,
            online=False,
            source=source,
            tags={
                "symbol": dataset_id.symbol,
                "timeframe": dataset_id.timeframe.value,
            },
        )

    # def get_feature_views(
    #     self,
    #     feature_df: DataFrame,
    #     feature_name: str,
    #     param_hash: str,
    #     sub_feature_names: List[str],
    #     asset_data: AssetPriceDataSet,
    #     symbol: str,
    # ) -> None:
    #     """
    #     Store computed features in the feature store.

    #     Args:
    #         feature_df: DataFrame containing the computed features
    #         feature_name: Name of the feature
    #         param_hash: Hash of the feature parameters
    #         sub_feature_names: List of sub-feature names in the feature
    #         asset_data: The asset price dataset containing metadata
    #         symbol: The trading symbol
    #     """
    #     if not self.feature_store or not self.config.enabled:
    #         return

    #     try:
    #         # Make a copy to avoid modifying the original dataframe
    #         store_df = feature_df.copy()

    #         # Add required columns for feast
    #         store_df["event_timestamp"] = store_df["Time"]
    #         store_df[self.config.entity_name] = self._get_entity_value(
    #             symbol, asset_data.timeframe.value
    #         )

    #         feature_view_name = self._get_feature_view_name(
    #             feature_name, param_hash, asset_data.timeframe.value
    #         )

    #         # Check if feature view exists
    #         feature_view_exists = True
    #         try:
    #             self.feature_store.get_feature_view(feature_view_name)
    #         except Exception:
    #             feature_view_exists = False
    #             logger.info(
    #                 f"Feature view {feature_view_name} does not exist, creating..."
    #             )

    #         # Create and apply feature view if needed
    #         if not feature_view_exists:
    #             # Create entity for this symbol and timeframe
    #             entity = self._get_entity(symbol, asset_data.timeframe.value)

    #             feature_view = self._create_feature_view(
    #                 feature_name=feature_name,
    #                 param_hash=param_hash,
    #                 sub_feature_names=sub_feature_names,
    #                 source_path=file_path,
    #                 entity=entity,
    #                 symbol=symbol,
    #                 timeframe=asset_data.timeframe.value,
    #             )

    #     except Exception as e:
    #         logger.warning(f"Failed to store features: {str(e)}")

    def create_feature_service(
        self,
        feature_views: list[FeatureView | OnDemandFeatureView],
        dataset_id: DatasetIdentifier,
        feature_version_info: FeatureConfigVersionInfo,
    ):

        return FeatureService(
            name=f"service_{dataset_id.symbol}_{dataset_id.timeframe.value}_v{feature_version_info.semver}-{feature_version_info.hash}",
            features=feature_views,
        )

    def get_entity(self, dataset_id: DatasetIdentifier) -> Entity:
        """
        Create an entity for the given dataset identifier.

        Args:
            dataset_id: Identifier for the dataset containing symbol and timeframe

        Returns:
            Entity: Feast entity for this symbol/timeframe combination
        """
        return Entity(
            name=self.feature_store_config.entity_name,
            join_keys=["symbol", "timeframe"],
            description=f"Entity for {dataset_id.symbol}{{{dataset_id.timeframe.value}}} asset price data",
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

    def _get_entity_key_formatted(self, dataset_id: DatasetIdentifier) -> str:
        """
        Get the entity value for the given dataset identifier.

        Args:
            dataset_id: Identifier for the dataset containing symbol and timeframe

        Returns:
            str: A unique identifier combining symbol and timeframe
        """
        return f"{dataset_id.symbol}_{dataset_id.timeframe.value}"
