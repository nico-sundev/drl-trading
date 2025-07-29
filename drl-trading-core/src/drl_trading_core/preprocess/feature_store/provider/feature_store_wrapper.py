
import os
from typing import Optional

from drl_trading_common.config.feature_config import FeatureStoreConfig
from feast import FeatureStore
from injector import inject


@inject
class FeatureStoreWrapper:

    _feature_store: Optional[FeatureStore] = None

    def __init__(self, feature_store_config: FeatureStoreConfig):
        self._feature_store_config = feature_store_config

    def get_feature_store(self) -> FeatureStore:
        """Get the underlying feature store instance."""
        if self._feature_store is None:
            config_directory = self._resolve_feature_store_config_directory()
            if config_directory is None:
                raise ValueError("Feature store is not enabled or config_directory is not configured")

            self._feature_store = FeatureStore(repo_path=config_directory)
        return self._feature_store

    def _resolve_feature_store_config_directory(self) -> Optional[str]:
        """
        Resolve the feature store configuration directory path based on STAGE.

        The config directory structure is:
        - base_config_directory/
          - dev/feature_store.yaml
          - cicd/feature_store.yaml
          - prod/feature_store.yaml
          - test/feature_store.yaml

        Returns:
            Optional[str]: Path to the stage-specific config directory, or None if not enabled
        """
        if not self._feature_store_config.enabled:
            return None

        stage = os.environ.get("STAGE")
        if stage is None:
            raise ValueError("STAGE environment variable must be set (dev, cicd, prod, test)")

        base_config_directory = self._feature_store_config.config_directory

        # Use absolute paths as-is
        if os.path.isabs(base_config_directory):
            base_resolved_path = base_config_directory
        else:
            # For relative paths, resolve against current working directory
            base_resolved_path = os.path.abspath(base_config_directory)

        # Append the stage subdirectory
        stage_config_directory = os.path.join(base_resolved_path, stage)

        # Verify the stage directory and feature_store.yaml exist
        feature_store_yaml = os.path.join(stage_config_directory, "feature_store.yaml")
        if not os.path.exists(feature_store_yaml):
            raise FileNotFoundError(f"Feature store config not found: {feature_store_yaml}")

        return stage_config_directory
