"""Common fixtures for integration tests."""

import pytest


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


@pytest.fixture(scope="session")
def docker_available() -> bool:
    """Fixture to check Docker availability for integration tests."""
    return is_docker_available()
