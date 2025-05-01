"""Unit tests for the dependency injection container."""

import os

import pytest

from ai_trading.config.application_config import ApplicationConfig
from ai_trading.config.config_loader import ConfigLoader
from ai_trading.config.environment_config import EnvironmentConfig
from ai_trading.config.feature_config import FeaturesConfig, FeatureStoreConfig
from ai_trading.config.local_data_import_config import LocalDataImportConfig
from ai_trading.config.rl_model_config import RlModelConfig
from ai_trading.data_import.base_data_import_service import BaseDataImportService
from ai_trading.data_import.local.csv_data_import_service import CsvDataImportService
from ai_trading.di.containers import ApplicationContainer


@pytest.fixture
def config():
    config = ConfigLoader.get_config(
        os.path.join(
            os.path.dirname(__file__), "../resources/applicationConfig-test.json"
        )
    )
    return config


@pytest.fixture
def container(config) -> ApplicationContainer:
    """Create a container initialized with the test configuration.

    Args:
        test_config_path: Path to the test configuration

    Returns:
        Configured ApplicationContainer instance
    """
    # Given
    container = ApplicationContainer(application_config=config)
    return container


def test_application_config_loads_successfully(container):
    """Test that the container can load application config successfully."""
    # Given
    # Container initialized with test config in fixture

    # When
    config = container.application_config()

    # Then
    assert isinstance(config, ApplicationConfig)
    assert hasattr(config, "features_config")
    assert hasattr(config, "local_data_import_config")
    assert hasattr(config, "rl_model_config")
    assert hasattr(config, "environment_config")
    assert hasattr(config, "feature_store_config")


def test_config_sections_are_accessible(container):
    """Test that individual config sections can be accessed separately."""
    # Given
    # Container initialized with test config

    # When
    features_config = container.features_config()
    local_data_import_config = container.local_data_import_config()
    rl_model_config = container.rl_model_config()
    environment_config = container.environment_config()
    feature_store_config = container.feature_store_config()

    # Then
    assert isinstance(features_config, FeaturesConfig)
    assert isinstance(local_data_import_config, LocalDataImportConfig)
    assert isinstance(rl_model_config, RlModelConfig)
    assert isinstance(environment_config, EnvironmentConfig)
    assert isinstance(feature_store_config, FeatureStoreConfig)


def test_config_sections_derive_from_main_config(container):
    """Test that config sections are derived from the main application config."""
    # Given
    # Container initialized with test config

    # When
    app_config = container.application_config()
    features_config = container.features_config()

    # Then
    # They should be the same object instance, not just equal
    assert features_config is app_config.features_config


def test_container_can_resolve_services(container):
    """Test that the container can resolve all registered services."""
    # Given
    # Container initialized with test config

    # When/Then
    # All these services should resolve without errors
    assert container.feature_class_registry()
    assert container.merge_service()
    assert container.strip_service()
    assert container.context_feature_service()
    assert container.csv_data_import_service()
    assert container.data_import_manager()
    assert container.feast_service()
    assert container.feature_aggregator()
    assert container.preprocess_service()
    assert container.split_service()
    assert container.agent_training_service()


def test_service_dependencies_use_config_sections(container):
    """Test that services receive the correct config section objects."""
    # Given
    # Container initialized with test config

    # When
    csv_service = container.csv_data_import_service()
    split_service = container.split_service()
    feature_aggregator = container.feature_aggregator()
    feast_service = container.feast_service()

    # Then
    # Verify services received proper config objects
    assert isinstance(csv_service.config, LocalDataImportConfig)
    assert isinstance(split_service.config, RlModelConfig)
    assert isinstance(feature_aggregator.config, FeaturesConfig)
    assert isinstance(feast_service.config, FeatureStoreConfig)


def test_data_import_manager_uses_abstract_interface(container):
    """Test that DataImportManager uses BaseDataImportService abstraction."""
    # Given
    # Container initialized with test config

    # When
    manager = container.data_import_manager()

    # Then
    assert isinstance(manager.import_service, BaseDataImportService)
    # Should specifically be the CSV implementation
    assert isinstance(manager.import_service, CsvDataImportService)


def test_singleton_services_are_reused(container):
    """Test that singleton services return the same instance when resolved multiple times."""
    # Given
    # Container initialized with test config

    # When
    instance1 = container.feature_class_registry()
    instance2 = container.feature_class_registry()

    # Then
    assert instance1 is instance2  # Same instance, not just equal


def test_config_sections_are_reused(container):
    """Test that config sections are cached and reused."""
    # Given
    # Container initialized with test config

    # When
    features_config1 = container.features_config()
    features_config2 = container.features_config()

    # Then
    assert features_config1 is features_config2  # Same instance, not just equal
