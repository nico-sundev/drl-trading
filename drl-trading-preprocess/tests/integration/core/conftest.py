import logging
import os
import shutil
import tempfile
from datetime import datetime
from typing import Generator

import pytest
from injector import Injector

from drl_trading_adapter.infrastructure.di.adapter_module import AdapterModule
from drl_trading_common.config.feature_config import (
    FeatureStoreConfig,
    LocalRepoConfig,
    S3RepoConfig,
)
from drl_trading_common.enum.offline_repo_strategy_enum import OfflineRepoStrategyEnum
from drl_trading_common.model.feature_config_version_info import (
    FeatureConfigVersionInfo,
)
from drl_trading_core.infrastructure.di.core_module import CoreModule
from drl_trading_preprocess.infrastructure.config.preprocess_config import PreprocessConfig
from drl_trading_preprocess.infrastructure.di.preprocess_module import PreprocessModule


@pytest.fixture(scope="session", autouse=True)
def register_cleanup_on_exit():
    """Register cleanup to run when Python exits - much simpler and more reliable.

    Since we use unique database files per test, we don't need complex session cleanup.
    The atexit handler runs when all file handles are guaranteed to be closed.
    """
    import atexit

    test_config_dir = os.path.join(os.path.dirname(__file__), "../../../resources/test")
    data_dir = os.path.join(test_config_dir, "data")

    # Register cleanup to run when Python process exits
    atexit.register(_cleanup_on_exit, data_dir)

    yield  # Let tests run


def _cleanup_on_exit(data_dir: str) -> None:
    """Clean up test data when Python exits - all file handles are closed by then."""
    try:
        if os.path.exists(data_dir):
            import shutil
            shutil.rmtree(data_dir, ignore_errors=True)
            print(f"DEBUG: Exit cleanup completed for: {data_dir}")
    except Exception as e:
        print(f"DEBUG: Exit cleanup best effort: {e}")  # Log but don't fail



@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Configure logging for integration tests with debug level."""
    # Set up logging configuration for integration tests
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # Override any existing logging configuration
    )

    # Set specific loggers to DEBUG level
    loggers = ["drl_trading_adapter", "drl_trading_common", "feast", "drl_trading_preprocess"]

    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

    # Also set the root logger
    logging.getLogger().setLevel(logging.DEBUG)

    print("DEBUG: Integration test logging configured to DEBUG level")


@pytest.fixture(scope="session")
def temp_feast_repo() -> Generator[str, None, None]:
    """Create a temporary directory for Feast repository during testing."""
    temp_dir = tempfile.mkdtemp(prefix="feast_integration_test_")
    yield temp_dir
    # Cleanup after all tests are done
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def feature_store_config(temp_feast_repo: str) -> FeatureStoreConfig:
    """Create a feature store configuration for integration tests."""

    # Create local repo config for the temporary feast repository
    local_repo_config = LocalRepoConfig(repo_path=str(temp_feast_repo))

    # Create S3 repo config for testing (optional, can be None for local strategy)
    s3_repo_config = S3RepoConfig(
        bucket_name="drl-trading-features-test",
        prefix="features",
        endpoint_url=None,
        region="us-east-1",
        access_key_id=None,
        secret_access_key=None,
    )

    # Set test config directory
    test_config_dir = os.path.join(os.path.dirname(__file__), "../../resources/test")

    return FeatureStoreConfig(
        cache_enabled=True,
        entity_name="test_entity",
        ttl_days=30,
        online_enabled=True,
        service_name="test-service",
        service_version="1.0.0",
        config_directory=test_config_dir,
        offline_repo_strategy=OfflineRepoStrategyEnum.LOCAL,
        local_repo_config=local_repo_config,
        s3_repo_config=s3_repo_config,
    )


@pytest.fixture(scope="function")
def real_feast_container(
    feature_store_config: FeatureStoreConfig,
    temp_feast_repo: str,
) -> Injector:
    """Create a dependency injection container with REAL Feast integration.

    This fixture provides a configured injector instance with real services
    for true integration testing. Uses minimal test features that don't depend
    on the strategy module, maintaining proper dependency boundaries.

    Args:
        test_feature_factory: Test feature factory with RSI and close price features
        config_fixture: Test configuration with temp Feast repository
        temp_feast_repo: Path to the temporary Feast repository

    Returns:
        Injector: Configured DI container with real services and test features
    """

    # Create injector with the real Feast configuration and CORE test modules only
    app_module = AdapterModule()
    core_module = CoreModule()
    preprocess_module = PreprocessModule(PreprocessConfig(feature_store_config=feature_store_config))
    injector = Injector([app_module, preprocess_module, core_module])

    return injector


@pytest.fixture
def feature_version_info_fixture() -> FeatureConfigVersionInfo:
    """Create feature version info for integration testing."""
    return FeatureConfigVersionInfo(
        semver="1.0.0",
        hash="integration_test_hash_123",
        created_at=datetime.fromisoformat("2024-01-01T00:00:00"),
        feature_definitions=[
            {"name": "rsi_14", "enabled": True, "role": "observation_space"},
            {"name": "close_1", "enabled": True, "role": "observation_space"},
            {"name": "reward", "enabled": True, "role": "reward_engineering"},
            {
                "name": "cumulative_return",
                "enabled": True,
                "role": "reward_engineering",
            },
        ],
    )


# Simple test fixtures for integration testing (no business logic)
@pytest.fixture
def clean_integration_environment() -> Generator[None, None, None]:
    """Clean integration test environment with proper isolation.

    Uses unique database paths per test to avoid file locking issues.
    This is the proper way to handle stateful resources in testing.
    """
    import uuid

    # Set STAGE environment variable for consistent testing
    os.environ.setdefault("STAGE", "test")

    # Use unique database paths per test to avoid file locking completely
    # This is the standard approach for database testing
    test_uuid = uuid.uuid4().hex[:8]
    os.environ["FEAST_REGISTRY_PATH"] = f"data/registry_{test_uuid}.db"
    os.environ["FEAST_ONLINE_STORE_PATH"] = f"data/online_store_{test_uuid}.db"

    yield

    # Clean up environment variables (files will be cleaned up by session fixture)
    for env_var in ["FEAST_REGISTRY_PATH", "FEAST_ONLINE_STORE_PATH", "FEAST_OFFLINE_STORE_PATH"]:
        os.environ.pop(env_var, None)
