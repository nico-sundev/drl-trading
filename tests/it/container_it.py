"""Unit tests for the dependency injection mocked_container."""

from drl_trading_framework.common.config.application_config import ApplicationConfig
from drl_trading_framework.common.config.environment_config import EnvironmentConfig
from drl_trading_framework.common.config.feature_config import (
    FeaturesConfig,
    FeatureStoreConfig,
)
from drl_trading_framework.common.config.local_data_import_config import (
    LocalDataImportConfig,
)
from drl_trading_framework.common.config.rl_model_config import RlModelConfig
from drl_trading_framework.common.data_import.base_data_import_service import (
    BaseDataImportService,
)
from drl_trading_framework.common.data_import.local.csv_data_import_service import (
    CsvDataImportService,
)


def test_application_config_loads_successfully(mocked_container):
    """Test that the container can load application config successfully."""
    # Given
    # Container initialized with test config in fixture

    # When
    config = mocked_container.application_config()

    # Then
    assert isinstance(config, ApplicationConfig)
    assert hasattr(config, "features_config")
    assert hasattr(config, "local_data_import_config")
    assert hasattr(config, "rl_model_config")
    assert hasattr(config, "environment_config")
    assert hasattr(config, "feature_store_config")


def test_config_sections_are_accessible(mocked_container):
    """Test that individual config sections can be accessed separately."""
    # Given
    # Container initialized with test config

    # When
    features_config = mocked_container.features_config()
    local_data_import_config = mocked_container.local_data_import_config()
    rl_model_config = mocked_container.rl_model_config()
    environment_config = mocked_container.environment_config()
    feature_store_config = mocked_container.feature_store_config()

    # Then
    assert isinstance(features_config, FeaturesConfig)
    assert isinstance(local_data_import_config, LocalDataImportConfig)
    assert isinstance(rl_model_config, RlModelConfig)
    assert isinstance(environment_config, EnvironmentConfig)
    assert isinstance(feature_store_config, FeatureStoreConfig)


def test_config_sections_derive_from_main_config(mocked_container):
    """Test that config sections are derived from the main application config."""
    # Given
    # Container initialized with test config

    # When
    app_config = mocked_container.application_config()
    features_config = mocked_container.features_config()

    # Then
    # They should be the same object instance, not just equal
    assert features_config is app_config.features_config


def test_container_can_resolve_services(mocked_container):
    """Test that the container can resolve all registered services."""
    # Given
    # Container initialized with test config

    # When/Then
    # All these services should resolve without errors
    assert mocked_container.feature_class_registry()
    assert mocked_container.merge_service()
    assert mocked_container.strip_service()
    assert mocked_container.context_feature_service()
    assert mocked_container.csv_data_import_service()
    assert mocked_container.data_import_manager()
    assert mocked_container.feast_service()
    assert mocked_container.feature_aggregator()
    assert mocked_container.preprocess_service()
    assert mocked_container.split_service()
    assert mocked_container.agent_training_service()


def test_service_dependencies_use_config_sections(mocked_container):
    """Test that services receive the correct config section objects."""
    # Given
    # Container initialized with test config

    # When
    csv_service = mocked_container.csv_data_import_service()
    split_service = mocked_container.split_service()
    feature_aggregator = mocked_container.feature_aggregator()
    feast_service = mocked_container.feast_service()

    # Then
    # Verify services received proper config objects
    assert isinstance(csv_service.config, LocalDataImportConfig)
    assert isinstance(split_service.config, RlModelConfig)
    assert isinstance(feature_aggregator.config, FeaturesConfig)
    assert isinstance(feast_service.config, FeatureStoreConfig)


def test_data_import_manager_uses_abstract_interface(mocked_container):
    """Test that DataImportManager uses BaseDataImportService abstraction."""
    # Given
    # Container initialized with test config

    # When
    manager = mocked_container.data_import_manager()

    # Then
    assert isinstance(manager.import_service, BaseDataImportService)
    # Should specifically be the CSV implementation
    assert isinstance(manager.import_service, CsvDataImportService)


def test_singleton_services_are_reused(mocked_container):
    """Test that singleton services return the same instance when resolved multiple times."""
    # Given
    # Container initialized with test config

    # When
    instance1 = mocked_container.feature_class_registry()
    instance2 = mocked_container.feature_class_registry()

    # Then
    assert instance1 is instance2  # Same instance, not just equal


def test_config_sections_are_reused(mocked_container):
    """Test that config sections are cached and reused."""
    # Given
    # Container initialized with test config

    # When
    features_config1 = mocked_container.features_config()
    features_config2 = mocked_container.features_config()

    # Then
    assert features_config1 is features_config2  # Same instance, not just equal
