import os
import subprocess
from pathlib import Path

import pytest
from feast import FeatureStore

from ai_trading.config.config_loader import ConfigLoader


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
