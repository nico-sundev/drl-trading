
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
        Resolve the feature store configuration directory path.

        This is the directory where feature_store.yaml is stored, separate from
        the actual data storage location which is configured per repository strategy.

        For relative paths, resolves against the current working directory.
        For absolute paths, uses the path as-is.

        Returns:
            Optional[str]: Path to the feature store configuration directory, or None if not enabled
        """
        if not self._feature_store_config.enabled:
            return None

        config_directory = self._feature_store_config.config_directory

        # Use absolute paths as-is
        if os.path.isabs(config_directory):
            resolved_path = config_directory
        else:
            # For relative paths, resolve against current working directory
            # This works for both development and test scenarios
            resolved_path = os.path.abspath(config_directory)

        if os.path.exists(resolved_path):
            yaml_path = os.path.join(resolved_path, "feature_store.yaml")
            print(f"DEBUG: feature_store.yaml exists: {os.path.exists(yaml_path)}")

        return resolved_path
