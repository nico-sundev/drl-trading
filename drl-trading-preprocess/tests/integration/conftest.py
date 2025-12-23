import logging
import os
import shutil
import tempfile
from typing import Generator

import pytest
from testcontainers.postgres import PostgresContainer

from drl_trading_common.config.feature_config import (
    FeatureStoreConfig,
    LocalRepoConfig,
    S3RepoConfig,
)
from drl_trading_common.config.infrastructure_config import DatabaseConfig
from drl_trading_common.enum.offline_repo_strategy_enum import OfflineRepoStrategyEnum


def is_docker_available() -> bool:
    """Check if Docker is available in the environment.

    Returns:
        bool: True if Docker daemon is accessible, False otherwise
    """
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def register_cleanup_on_exit() -> Generator[None, None, None]:
    """Register cleanup to run when Python exits - much simpler and more reliable.

    Since we use unique database files per test, we don't need complex session cleanup.
    The atexit handler runs when all file handles are guaranteed to be closed.
    """
    import atexit

    test_config_dir = os.path.join(os.path.dirname(__file__), "../resources/test")
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
def configure_logging() -> None:
    """Configure logging for integration tests with debug level."""
    # Set up logging configuration for integration tests
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # Override any existing logging configuration
    )

    # Set specific loggers to DEBUG level
    loggers = [
        "drl_trading_adapter",
        "drl_trading_common",
        "feast",
        "drl_trading_preprocess",
    ]

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
    test_config_dir = os.path.join(os.path.dirname(__file__), "../resources/test")

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
    for env_var in [
        "FEAST_REGISTRY_PATH",
        "FEAST_ONLINE_STORE_PATH",
        "FEAST_OFFLINE_STORE_PATH",
    ]:
        os.environ.pop(env_var, None)


@pytest.fixture(scope="function")
def postgres_container() -> Generator[PostgresContainer, None, None]:
    """Start PostgreSQL/TimescaleDB container for integration tests.

    Uses scope="function" to ensure each test gets a fresh database.
    This prevents test pollution and makes tests truly independent.
    Skips if Docker is not available (e.g., in CI environments without Docker).
    """
    if not is_docker_available():
        pytest.skip("Docker is not available - skipping PostgreSQL container tests")

    with PostgresContainer("postgres:15") as postgres:
        yield postgres


@pytest.fixture(scope="function")
def database_config(postgres_container: PostgresContainer) -> DatabaseConfig:
    """Create database configuration from testcontainer.

    Note: This fixture depends on postgres_container which requires Docker.

    Args:
        postgres_container: Running PostgreSQL container

    Returns:
        DatabaseConfig with connection details from container
    """
    return DatabaseConfig(
        provider="postgresql",
        host=postgres_container.get_container_host_ip(),
        port=postgres_container.get_exposed_port(5432),
        database=postgres_container.dbname,
        username=postgres_container.username,
        password=postgres_container.password,
    )
