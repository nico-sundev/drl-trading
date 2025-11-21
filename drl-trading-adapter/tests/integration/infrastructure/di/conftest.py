import pytest
from injector import Injector

from drl_trading_adapter.infrastructure.di.adapter_module import AdapterModule
from drl_trading_common.config.feature_config import FeatureStoreConfig, LocalRepoConfig
from drl_trading_common.enum.offline_repo_strategy_enum import OfflineRepoStrategyEnum


@pytest.fixture(scope="session")
def feature_store_config() -> FeatureStoreConfig:
    """Create a feature store configuration for integration tests."""
    return FeatureStoreConfig(
        cache_enabled=True,
        entity_name="test_entity",
        ttl_days=30,
        online_enabled=False,
        service_name="test-service",
        service_version="1.0.0",
        config_directory="tests/resources/feature_store",
        offline_repo_strategy=OfflineRepoStrategyEnum.LOCAL,
        local_repo_config=LocalRepoConfig(repo_path="tests/resources/features"),
    )


@pytest.fixture(scope="session")
def mocked_container(feature_store_config: FeatureStoreConfig) -> Injector:
    """Create a dependency injection container for integration tests."""
    app_module = AdapterModule()
    injector = Injector([app_module])

    # Override the feature factory with our test fixture
    injector.binder.bind(FeatureStoreConfig, to=feature_store_config)

    return injector
