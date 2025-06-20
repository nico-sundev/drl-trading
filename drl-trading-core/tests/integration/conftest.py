import os
import shutil
import subprocess
from pathlib import Path

import pytest
from drl_trading_common.config.application_config import ApplicationConfig
from drl_trading_common.config.config_loader import ConfigLoader
from drl_trading_strategy.feature.feature_factory import (
    FeatureFactoryInterface,
)
from feast import FeatureStore
from injector import Injector


@pytest.fixture(scope="session")
def mocked_config() -> ApplicationConfig:
    """Load test configuration from the test resources directory."""
    config_path = os.path.join(
        os.path.dirname(__file__), "../resources/applicationConfig-test.json"
    )
    return ConfigLoader.get_config(ApplicationConfig, path=config_path)


@pytest.fixture(scope="session")
def mocked_feature_store(request, mocked_config: ApplicationConfig) -> FeatureStore:
    """Initialize the Feast feature store for testing.

    Sets up a clean feature store for testing and ensures proper cleanup
    after the test session completes.

    Args:
        request: pytest request object
        mocked_config: The test configuration fixture

    Returns:
        FeatureStore: Initialized feature store instance
    """
    # Get paths from config
    repo_path = mocked_config.feature_store_config.repo_path
    store_path = mocked_config.feature_store_config.offline_store_path

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if not os.path.isabs(repo_path):
        abs_repo_path = os.path.join(project_root, repo_path)
    else:
        abs_repo_path = repo_path

    if not os.path.isabs(store_path):
        abs_store_path = os.path.join(project_root, store_path)
    else:
        abs_store_path = store_path

    # Clean up any existing feature store data before starting
    _clean_feature_store(abs_repo_path, abs_store_path)

    # Create data directory within repo path
    data_dir = os.path.join(abs_repo_path, "data")
    Path(data_dir).mkdir(exist_ok=True)

    # Initialize the repository
    try:
        subprocess.run(
            ["feast", "apply"], cwd=abs_repo_path, check=True, capture_output=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error initializing feast: {e.stderr.decode()}")
        raise

    # Return the FeatureStore instance
    feature_store = FeatureStore(repo_path=abs_repo_path)

    # Clean up function for after the session
    def cleanup():
        """Remove feature store data after test session completes."""
        _clean_feature_store(abs_repo_path, abs_store_path)
        print("Feature store cleaned up successfully after test session.")

    request.addfinalizer(cleanup)
    return feature_store


def _clean_feature_store(repo_path: str, store_path: str) -> None:
    """Helper function to clean up feature store data.

    Args:
        repo_path: Path to the feature store repository
        store_path: Path to the feature store data directory
    """
    # Clean the registry DB
    registry_db = os.path.join(repo_path, "data", "registry.db")
    if os.path.exists(registry_db):
        os.remove(registry_db)
        print(f"Removed feature store registry: {registry_db}")

    # Clean the feature store data directory
    if os.path.exists(store_path):
        shutil.rmtree(store_path)
        print(f"Removed feature store data directory: {store_path}")
        # Recreate the empty directory
        os.makedirs(store_path, exist_ok=True)

@pytest.fixture(scope="session")
def mocked_container(feature_factory, mocked_feature_store):
    """Create a mocked dependency injection container using the injector library.

    This fixture provides a configured injector instance with test dependencies
    for integration testing, replacing the legacy dependency-injector approach.
    """
    # Import necessary types for the injector setup

    # Use the test config path
    test_config_path = os.path.join(
        os.path.dirname(__file__), "../resources/applicationConfig-test.json"
    )    # Import and create the application module
    from drl_trading_core.common.di.core_module import CoreModule

    # Create injector with the domain module
    app_module = CoreModule(config_path=test_config_path)
    injector = Injector([app_module])

    # Override the feature factory with our test fixture
    injector.binder.bind(FeatureFactoryInterface, to=feature_factory)

    # Override the feature store with our test fixture
    injector.binder.bind(FeatureStore, to=mocked_feature_store)

    return injector
