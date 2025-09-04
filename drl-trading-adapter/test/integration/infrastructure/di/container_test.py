"""Test the mocked_container fixture."""

from drl_trading_adapter.adapter.feature_store.offline.offline_feature_repo_interface import (
    IOfflineFeatureRepository,
)
from drl_trading_adapter.infrastructure.di.adapter_module import AdapterModule


def test_injector_loads_successfully(mocked_container):
    """Test that the mocked_container can load application config successfully."""
    # When: We get the application config from the injector
    _ = mocked_container.get(AdapterModule)
    offline_feature_repo = mocked_container.get(IOfflineFeatureRepository)

    # Then: It should be properly loaded
    assert isinstance(offline_feature_repo, IOfflineFeatureRepository)
