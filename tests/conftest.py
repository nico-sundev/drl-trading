import os
import subprocess
from pathlib import Path

import pytest
from feast import FeatureStore

from ai_trading.config.config_loader import ConfigLoader
from ai_trading.di.containers import ApplicationContainer


@pytest.fixture(scope="session")
def test_config():
    """Load test configuration from the test resources directory."""
    config_path = os.path.join(
        os.path.dirname(__file__), "resources/applicationConfig-test.json"
    )
    return ConfigLoader.get_config(config_path)


@pytest.fixture(scope="session")
def test_container(test_config):
    """Create a container initialized with the test configuration.

    This fixture provides a configured ApplicationContainer instance
    that can be used across all tests.

    Args:
        test_config: The test configuration fixture

    Returns:
        Configured ApplicationContainer instance
    """
    container = ApplicationContainer(application_config=test_config)
    return container


@pytest.fixture(scope="session")
def init_feast_for_tests(request):
    """Initialize the Feast feature store for testing."""
    # Load test config
    config_path = os.path.join(
        os.path.dirname(__file__), "resources/applicationConfig-test.json"
    )
    config = ConfigLoader.get_config(config_path)

    # Get paths from config
    repo_path = config.feature_store_config.repo_path
    store_path = config.feature_store_config.offline_store_path

    # Create data directory within repo path if it doesn't exist
    data_dir = os.path.join(repo_path, "data")
    Path(data_dir).mkdir(exist_ok=True)

    # Initialize the repository if not already set up
    if not os.path.exists(os.path.join(repo_path, "data/registry.db")):
        # Apply the feature definitions to create the feature store
        try:
            subprocess.run(
                ["feast", "apply"], cwd=repo_path, check=True, capture_output=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error initializing feast: {e.stderr.decode()}")
            raise

    # Return the FeatureStore instance
    feature_store = FeatureStore(repo_path=repo_path)

    # Clean up function
    def cleanup():
        # Optional: clean up the registry after tests complete
        pass

    request.addfinalizer(cleanup)
    return feature_store
