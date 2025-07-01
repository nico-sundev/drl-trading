
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

    def get_feature_store(self):
        """Get the underlying feature store instance."""
        if self._feature_store is None:
            self._feature_store = FeatureStore(repo_path=self._resolve_feature_store_path())
        return self._feature_store

    def _resolve_feature_store_path(self) -> Optional[str]:
        """
        Resolve the feature store path based on configuration.
        If the path is relative, it will be resolved against the project root directory.

        Returns:
            Optional[str]: Absolute path to the feature store repository, or None if not enabled
        """
        if not self._feature_store_config.enabled:
            return None

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        )
        if not os.path.isabs(self._feature_store_config.repo_path):
            abs_file_path = os.path.join(
                project_root, self._feature_store_config.repo_path
            )
        else:
            abs_file_path = self._feature_store_config.repo_path
        return abs_file_path
