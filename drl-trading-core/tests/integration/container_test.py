"""Test the mocked_container fixture."""

from drl_trading_common.config.application_config import ApplicationConfig

from drl_trading_core.common.di.core_module import CoreModule
from drl_trading_core.preprocess.feature.feature_aggregator import (
    FeatureAggregator,
    FeatureAggregatorInterface,
)


def test_injector_loads_successfully(mocked_container):
    """Test that the mocked_container can load application config successfully."""
    # When: We get the application config from the injector
    module = mocked_container.get(CoreModule)
    config = mocked_container.get(ApplicationConfig)
    feature_aggregator = mocked_container.get(FeatureAggregatorInterface)

    # Then: It should be properly loaded
    assert isinstance(config, ApplicationConfig)
    assert hasattr(config, "features_config")
    assert hasattr(config, "local_data_import_config")
    assert hasattr(config, "rl_model_config")
    assert hasattr(config, "environment_config")
    assert hasattr(config, "feature_store_config")

    assert isinstance(feature_aggregator, FeatureAggregator)
